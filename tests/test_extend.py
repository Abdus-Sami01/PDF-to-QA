import json
from pathlib import Path

import pytest

from pdfqa import registry
from pdfqa.chunking import chunk_tree
from pdfqa.cli import main
from pdfqa.config import Config
from pdfqa.export import export
from pdfqa.extract import load
from pdfqa.graph import build_graph
from pdfqa.pipeline import Pipeline, supported
from pdfqa.plan import collect_chunks, estimate, format_plan
from pdfqa.records import Provenance, QARecord

FIXTURES = Path(__file__).parent / "fixtures"
PLUGIN = FIXTURES / "sample_plugin.py"


@pytest.fixture
def clean_registry():
    registry.clear()
    yield registry
    registry.clear()


@pytest.fixture
def plugin(clean_registry):
    registry.load_plugins([str(PLUGIN)])
    return registry


# ------------------------------------------------------------------ registry mechanics


def test_nothing_is_registered_by_default(clean_registry):
    assert registry.summary() == {"tasks": [], "formats": [], "adapters": [], "plugins": []}


def test_loading_a_plugin_registers_all_three_kinds(plugin):
    summary = registry.summary()
    assert summary["tasks"] == ["definition"]
    assert summary["formats"] == ["minimal"]
    assert summary["adapters"] == [".log"]


def test_a_plugin_is_only_loaded_once(plugin):
    assert registry.load_plugins([str(PLUGIN)]) == []


def test_loading_an_unknown_plugin_raises(clean_registry):
    with pytest.raises((ImportError, ModuleNotFoundError)):
        registry.load_plugins(["pdfqa_nonexistent_plugin_module"])


def test_decorators_can_be_used_directly(clean_registry):
    @registry.register_task("inline")
    def _task(runtime, chunks, kg, config):
        return []

    @registry.register_format("inline")
    def _fmt(records):
        return []

    assert registry.TASKS["inline"] is _task
    assert registry.FORMATS["inline"] is _fmt


# ------------------------------------------------------------------ custom adapter


def test_a_registered_suffix_becomes_loadable(plugin, tmp_path):
    log = tmp_path / "server.log"
    log.write_text("routing started\nselected 4 of 32 experts\n")
    tree = load(log)
    assert tree.meta["format"] == "log"
    assert "selected 4 of 32 experts" in tree.root.text_content()


def test_registered_suffixes_join_directory_expansion(plugin):
    assert ".log" in supported()


def test_unsupported_suffix_message_lists_plugin_suffixes(plugin, tmp_path):
    bad = tmp_path / "x.rtf"
    bad.write_text("hi")
    with pytest.raises(ValueError, match=r"\.log"):
        load(bad)


# ------------------------------------------------------------------ custom format


def test_a_registered_format_is_written(plugin, tmp_path):
    records = [QARecord(question="q1", answer="a1", context="c", prov=Provenance(source="s"))]
    written = export(records, tmp_path, ["minimal"])
    rows = [json.loads(l) for l in Path(written["minimal"]).read_text().splitlines()]
    assert rows == [{"q": "q1", "a": "a1"}]


def test_an_unregistered_format_still_errors(clean_registry, tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        export([], tmp_path, ["nonsense"])


# ------------------------------------------------------------------ custom task


def test_a_registered_task_contributes_records(plugin, tree, runtime):
    chunks = chunk_tree(tree, max_tokens=300)
    kg = build_graph(chunks, runtime)
    cfg = Config()
    produced = registry.TASKS["definition"](runtime, chunks, kg, cfg)
    assert produced and all(r.task == "definition" for r in produced)


def test_the_pipeline_runs_registered_tasks(config, clean_registry):
    config.plugins = [str(PLUGIN)]
    config.formats = ["raw", "minimal"]
    report = Pipeline(config).run()
    assert report["plugins"] == [str(PLUGIN)]
    assert report["registered"]["tasks"] == ["definition"]
    assert Path(config.outdir, "minimal.jsonl").exists()


def test_selecting_an_unknown_task_fails_loudly(config, plugin):
    config.synth.tasks = ["does_not_exist"]
    with pytest.raises(ValueError, match="unknown task"):
        Pipeline(config).run()


def test_builtin_tasks_are_unaffected_by_an_empty_registry(config, clean_registry):
    report = Pipeline(config).run()
    assert report["stats"]["count"] > 0
    assert "plugins" not in report


# ------------------------------------------------------------------ plan


def test_plan_counts_calls_without_contacting_a_model(runtime):
    cfg = Config()
    chunks = collect_chunks(cfg, [FIXTURES / "sample.md", FIXTURES / "sample2.md"])
    before = len(runtime.calls)
    plan = estimate(cfg, chunks)
    assert len(runtime.calls) == before
    assert plan.totals()["calls"] > 0
    assert plan.chunks == sum(len(v) for v in chunks.values())


def test_plan_attributes_calls_to_roles():
    cfg = Config()
    plan = estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"]))
    roles = plan.by_role()
    assert roles["verify"]["calls"] > 0
    assert roles["generate"]["calls"] > 0


def test_verification_dominates_the_projected_cost():
    cfg = Config()
    plan = estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"]))
    verify = next(s for s in plan.stages if s.stage == "verify")
    assert verify.calls > plan.totals()["calls"] / 2


def test_disabling_gates_removes_them_from_the_plan():
    cfg = Config()
    cfg.verify.model_gates = False
    cfg.graph.llm_extract = False
    plan = estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"]))
    stages = {s.stage for s in plan.stages}
    assert "verify" not in stages and "graph" not in stages


def test_cross_document_is_only_planned_for_multiple_documents():
    cfg = Config()
    one = estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"]))
    two = estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md", FIXTURES / "sample2.md"]))
    assert "cross_document" not in {s.stage for s in one.stages}
    assert "cross_document" in {s.stage for s in two.stages}


def test_figure_qa_is_only_planned_with_a_vision_backend():
    cfg = Config()
    assert "figure_qa" not in {s.stage for s in estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"])).stages}
    cfg.runtime.vision = {"backend": "echo"}
    assert "figure_qa" in {s.stage for s in estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"])).stages}


def test_cost_scales_with_the_prices_given():
    cfg = Config()
    plan = estimate(cfg, collect_chunks(cfg, [FIXTURES / "sample.md"]))
    assert plan.cost(0.0, 0.0) == 0.0
    assert plan.cost(6.0, 30.0) == pytest.approx(2 * plan.cost(3.0, 15.0))


def test_empty_input_plans_nothing():
    plan = estimate(Config(), {})
    assert plan.totals()["calls"] == 0
    assert "documents: 0" in format_plan(plan)


def test_plan_command_prints_a_table(capsys):
    assert main(["plan", str(FIXTURES / "sample.md"), "--price-in", "3", "--price-out", "15"]) == 0
    out = capsys.readouterr().out
    assert "estimated cost" in out and "verify" in out


def test_plan_command_without_inputs_fails(capsys):
    assert main(["plan"]) == 1
    assert "no inputs" in capsys.readouterr().err
