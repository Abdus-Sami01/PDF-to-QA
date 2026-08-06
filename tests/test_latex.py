import pytest

from pdfqa.chunking import chunk_tree
from pdfqa.docast import EQUATION
from pdfqa.extract import from_markdown
from pdfqa.latex import Node, depth, environments, parse, tokenize, validate


def issues(source):
    return validate(source)["issues"]


def test_tokenizer_splits_commands_numbers_and_scripts():
    kinds = [(t.kind, t.value) for t in tokenize(r"x^2 + \alpha_{i}")]
    assert ("command", "alpha") in kinds
    assert ("sup", "^") in kinds and ("sub", "_") in kinds
    assert ("number", "2") in kinds


def test_a_well_formed_equation_parses_clean():
    result = validate(r"s_i = x^\top W_r e_i + b_i \quad (1)")
    assert result["valid"] and result["issues"] == []
    assert "top" in result["symbols"] and "W" in result["symbols"]
    assert result["depth"] >= 2


def test_fraction_arguments_become_children():
    tree, errs = parse(r"\frac{a}{b}")
    assert errs == []
    frac = tree.children[0]
    assert frac.kind == "command" and frac.value == "frac"
    assert len(frac.children) == 2


def test_missing_fraction_argument_is_reported():
    assert any("frac" in i for i in issues(r"\frac{a}"))


def test_unclosed_brace_is_reported():
    assert any("missing closing" in i for i in issues(r"\sqrt{x + 1"))


def test_script_without_an_operand_is_reported():
    assert any("no operand" in i for i in issues("x^"))


def test_script_without_a_base_is_reported():
    assert any("no base" in i for i in issues("^2 + x"))


def test_truncated_equation_is_caught_by_dangling_operator():
    assert any("ends with operator" in i for i in issues("a + b ="))
    assert issues("a + b = c") == []


def test_environment_mismatch_is_reported():
    assert environments(r"\begin{aligned} x \end{matrix}")
    assert environments(r"\begin{aligned} x \end{aligned}") == []
    assert any("never closed" in i for i in environments(r"\begin{aligned} x"))


def test_environments_do_not_pollute_symbols():
    result = validate(r"\begin{aligned} x &= 1 \end{aligned}")
    assert result["valid"]
    assert result["symbols"] == ["x"]


def test_unknown_command_is_flagged_but_single_letter_escapes_are_not():
    assert any("unknown command" in i for i in issues(r"\foo{x}"))
    assert not any("unknown command" in i for i in issues(r"\, x"))


def test_empty_equation_is_invalid():
    assert issues("") == ["empty equation"]
    assert issues("   ") == ["empty equation"]


def test_equation_number_is_stripped_before_validation():
    assert validate("a = b (1)")["valid"]


def test_depth_reflects_nesting():
    assert depth(Node("seq", children=[Node("group", children=[Node("ident", "x")])])) == 3


def test_extracted_equations_carry_a_tree(sample_text):
    tree = from_markdown(sample_text, source="sample.md")
    eq = tree.nodes([EQUATION])[0]
    assert eq.attrs["balanced"] is True
    assert eq.attrs["tree"]["kind"] == "seq"
    assert eq.attrs["depth"] >= 2
    assert "issues" not in eq.attrs


def test_malformed_equation_survives_extraction_with_its_reasons():
    tree = from_markdown("# T\n\n## S\n\n$$\n\\frac{a}{\n$$\n", source="bad.md")
    eq = tree.nodes([EQUATION])[0]
    assert eq.attrs["balanced"] is False
    assert eq.attrs["issues"]
    assert chunk_tree(tree)


def test_equation_attrs_round_trip_through_the_tree(sample_text):
    from pdfqa.docast import DocumentTree

    tree = from_markdown(sample_text, source="sample.md")
    clone = DocumentTree.from_dict(tree.as_dict())
    assert clone.nodes([EQUATION])[0].attrs["tree"] == tree.nodes([EQUATION])[0].attrs["tree"]
