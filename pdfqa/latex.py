"""A small LaTeX math parser: tokens to a syntax tree, plus structural validation.

Malformed equations are usually an extraction defect rather than an authoring one — a dropped
superscript or a swallowed brace means the chunk feeding the generator is already wrong. Parsing
turns that from an invisible problem into a flag with a reason attached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COMMAND = re.compile(r"\\([A-Za-z]+|.)")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
IDENT = re.compile(r"[A-Za-z]")
SPACE = re.compile(r"\s+")

OPENERS = {"{": "}", "(": ")", "[": "]"}
CLOSERS = {v: k for k, v in OPENERS.items()}
BINARY = set("+-*/=<>") | {"\\pm", "\\times", "\\div", "\\cdot", "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv", "\\propto"}

KNOWN = {
    "frac", "sqrt", "sum", "prod", "int", "lim", "log", "ln", "exp", "sin", "cos", "tan", "max", "min",
    "argmax", "argmin", "top", "cdot", "times", "div", "pm", "mp", "leq", "geq", "neq", "approx", "equiv",
    "propto", "in", "notin", "subset", "subseteq", "forall", "exists", "partial", "nabla", "infty",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "varphi", "chi",
    "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi",
    "Omega", "mathbb", "mathcal", "mathbf", "mathrm", "text", "textrm", "operatorname", "left", "right",
    "begin", "end", "quad", "qquad", "hat", "bar", "tilde", "vec", "dot", "ddot", "overline", "underline",
    "big", "Big", "bigg", "Bigg", "langle", "rangle", "lfloor", "rfloor", "lceil", "rceil", "label",
}

ARITY = {"frac": 2, "sqrt": 1, "hat": 1, "bar": 1, "tilde": 1, "vec": 1, "overline": 1, "underline": 1,
         "mathbb": 1, "mathcal": 1, "mathbf": 1, "mathrm": 1, "text": 1, "textrm": 1, "operatorname": 1}


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if (m := SPACE.match(source, i)) :
            i = m.end()
            continue
        if ch == "\\":
            m = COMMAND.match(source, i)
            if m is None:
                tokens.append(Token("text", ch, i))
                i += 1
                continue
            tokens.append(Token("newline" if m.group(1) == "\\" else "command", m.group(1), i))
            i = m.end()
            continue
        if ch in OPENERS:
            tokens.append(Token("open", ch, i))
        elif ch in CLOSERS:
            tokens.append(Token("close", ch, i))
        elif ch == "_":
            tokens.append(Token("sub", ch, i))
        elif ch == "^":
            tokens.append(Token("sup", ch, i))
        elif ch == "&":
            tokens.append(Token("align", ch, i))
        elif (m := NUMBER.match(source, i)) :
            tokens.append(Token("number", m.group(0), i))
            i = m.end()
            continue
        elif IDENT.match(ch):
            tokens.append(Token("ident", ch, i))
        else:
            tokens.append(Token("op" if ch in "+-*/=<>|,.;:!" else "text", ch, i))
        i += 1
    return tokens


@dataclass
class Node:
    kind: str
    value: str = ""
    children: list["Node"] = field(default_factory=list)

    def as_dict(self) -> dict:
        out: dict = {"kind": self.kind}
        if self.value:
            out["value"] = self.value
        if self.children:
            out["children"] = [c.as_dict() for c in self.children]
        return out

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0
        self.issues: list[str] = []

    def peek(self) -> Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def next(self) -> Token | None:
        tok = self.peek()
        if tok is not None:
            self.i += 1
        return tok

    def parse(self) -> Node:
        root = Node("seq", children=self.sequence(stop=None))
        if self.peek() is not None:
            tok = self.peek()
            self.issues.append(f"unmatched '{tok.value}' at {tok.pos}")
        return root

    def sequence(self, stop: str | None) -> list[Node]:
        out: list[Node] = []
        while (tok := self.peek()) is not None:
            if tok.kind == "close":
                if stop is not None and tok.value == stop:
                    return out
                return out
            out.append(self.atom_with_scripts(out))
        if stop is not None:
            self.issues.append(f"missing closing '{stop}'")
        return out

    def atom_with_scripts(self, siblings: list[Node]) -> Node:
        base = self.atom()
        while (tok := self.peek()) is not None and tok.kind in ("sub", "sup"):
            self.next()
            if base is None or base.kind == "op":
                self.issues.append(f"'{tok.value}' has no base at {tok.pos}")
            operand = self.atom()
            if operand is None:
                self.issues.append(f"'{tok.value}' has no operand at {tok.pos}")
                operand = Node("missing")
            base = Node("script" if tok.kind == "sup" else "subscript", children=[base or Node("missing"), operand])
        return base or Node("missing")

    def atom(self) -> Node | None:
        tok = self.next()
        if tok is None:
            return None
        if tok.kind == "open":
            children = self.sequence(stop=OPENERS[tok.value])
            closing = self.peek()
            if closing is not None and closing.kind == "close" and closing.value == OPENERS[tok.value]:
                self.next()
            else:
                self.issues.append(f"missing closing '{OPENERS[tok.value]}' for '{tok.value}' at {tok.pos}")
            return Node("group", tok.value, children)
        if tok.kind == "command":
            return self.command(tok)
        if tok.kind in ("sub", "sup"):
            self.issues.append(f"'{tok.value}' has no base at {tok.pos}")
            return Node("missing")
        return Node(tok.kind, tok.value)

    def command(self, tok: Token) -> Node:
        name = tok.value
        if name not in KNOWN and len(name) > 1:
            self.issues.append(f"unknown command '\\{name}' at {tok.pos}")
        args = []
        for _ in range(ARITY.get(name, 0)):
            nxt = self.peek()
            if nxt is None or nxt.kind == "close":
                self.issues.append(f"'\\{name}' is missing an argument")
                args.append(Node("missing"))
                continue
            arg = self.atom()
            args.append(arg or Node("missing"))
        return Node("command", name, args)


BEGIN_END = re.compile(r"\\(begin|end)\s*\{([^}]*)\}")


def environments(source: str) -> list[str]:
    stack: list[str] = []
    issues: list[str] = []
    for m in BEGIN_END.finditer(source):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif not stack:
            issues.append(f"\\end{{{m.group(2)}}} with no matching \\begin")
        elif stack[-1] != m.group(2):
            issues.append(f"\\end{{{m.group(2)}}} closes \\begin{{{stack.pop()}}}")
        else:
            stack.pop()
    issues.extend(f"\\begin{{{name}}} was never closed" for name in stack)
    return issues


def parse(source: str) -> tuple[Node, list[str]]:
    parser = Parser(tokenize(source))
    tree = parser.parse()
    return tree, parser.issues + environments(source)


def dangling_operators(tree: Node) -> list[str]:
    """A binary operator at either end usually means the extractor cut the equation in half."""
    terms = [c for c in tree.children if c.kind != "text"]
    issues = []
    if terms and terms[0].kind == "op" and terms[0].value in "*/=<>":
        issues.append(f"equation starts with operator '{terms[0].value}'")
    if terms and terms[-1].kind == "op" and terms[-1].value in "+-*/=<>":
        issues.append(f"equation ends with operator '{terms[-1].value}'")
    return issues


def validate(source: str) -> dict:
    """Returns the syntax tree plus everything structurally wrong with it."""
    raw = source or ""
    body = re.sub(r"\\label\s*\{[^}]*\}", "", raw).strip()
    body = re.sub(r"\((\d{1,3})\)\s*$", "", body).strip()
    env_issues = environments(body)
    body = BEGIN_END.sub("", body).strip()
    if not body:
        return {"valid": False, "issues": ["empty equation"] + env_issues, "tree": None, "symbols": [], "depth": 0}

    tree, issues = parse(body)
    issues = [i for i in issues if not i.startswith("\\begin") and not i.startswith("\\end")] + env_issues
    issues = issues + dangling_operators(tree)
    if not tree.children:
        issues.append("no terms parsed")
    return {
        "valid": not issues,
        "issues": issues,
        "tree": tree.as_dict(),
        "symbols": sorted({n.value for n in tree.walk() if n.kind in ("ident", "command") and n.value}),
        "depth": depth(tree),
    }


def depth(node: Node) -> int:
    return 1 + max((depth(c) for c in node.children), default=0)
