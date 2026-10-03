"""Boolean query parser.

Grammar (recursive descent, standard precedence ``NOT`` > ``AND`` > ``OR``)::

    expression := or_expr
    or_expr    := and_expr ( OR and_expr )*
    and_expr   := unary ( AND unary )*
    unary      := NOT unary | primary
    primary    := '(' expression ')' | PHRASE | TERM

The parser builds an AST of small dataclasses. Nothing is ever handed to
``eval``/``exec``: the AST is interpreted by :mod:`src.phase3.boolean_search`.

``NOT`` is deliberately restricted to ``A AND NOT B`` (with parentheses for
compound right-hand sides). A bare ``NOT term`` would return almost the whole
corpus, which is meaningless for retrieval, so it is rejected with a message
that says what to write instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

#: Query operators.
OPERATORS = ("AND", "OR", "NOT")


class QuerySyntaxError(ValueError):
    """Raised for a malformed Boolean query."""


# ----------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Term:
    """A single term or quoted phrase in the query."""

    text: str
    is_phrase: bool = False


@dataclass(frozen=True)
class Not:
    """Negation of a sub-expression (only valid inside an AND)."""

    child: "Expression"


@dataclass(frozen=True)
class And:
    left: "Expression"
    right: "Expression"


@dataclass(frozen=True)
class Or:
    left: "Expression"
    right: "Expression"


Expression = Union[Term, Not, And, Or]


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class QueryToken:
    kind: str          # TERM | PHRASE | AND | OR | NOT | LPAREN | RPAREN
    text: str


_TOKEN_RE = re.compile(
    r"""
    (?P<phrase>"[^"]*")
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<word>[^\s()"]+)
    """,
    re.VERBOSE,
)


def tokenize_query(query: str) -> List[QueryToken]:
    """Split a query into tokens, recognising quoted phrases and operators."""
    tokens: List[QueryToken] = []
    for match in _TOKEN_RE.finditer(query or ""):
        kind = match.lastgroup
        text = match.group()
        if kind == "phrase":
            inner = text[1:-1].strip()
            if not inner:
                raise QuerySyntaxError(f"empty quoted phrase in query: {query!r}")
            tokens.append(QueryToken("PHRASE", inner))
        elif kind == "lparen":
            tokens.append(QueryToken("LPAREN", text))
        elif kind == "rparen":
            tokens.append(QueryToken("RPAREN", text))
        else:
            upper = text.upper()
            if upper in OPERATORS:
                tokens.append(QueryToken(upper, upper))
            else:
                tokens.append(QueryToken("TERM", text))
    if not tokens:
        raise QuerySyntaxError(f"query is empty: {query!r}")
    return tokens


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
class QueryParser:
    """Recursive-descent parser producing an :data:`Expression` AST."""

    def __init__(self, query: str, allow_parentheses: bool = True,
                 allow_not: bool = True) -> None:
        self.query = query
        self.tokens = tokenize_query(query)
        self.position = 0
        self.allow_parentheses = allow_parentheses
        self.allow_not = allow_not

    # ---------------- helpers ----------------
    def peek(self) -> Optional[QueryToken]:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def next(self) -> QueryToken:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError(f"unexpected end of query: {self.query!r}")
        self.position += 1
        return token

    def expect(self, kind: str) -> QueryToken:
        token = self.next()
        if token.kind != kind:
            raise QuerySyntaxError(
                f"expected {kind} but found {token.kind} ({token.text!r}) in query {self.query!r}"
            )
        return token

    # ---------------- grammar ----------------
    def parse(self) -> Expression:
        expression = self.parse_or()
        leftover = self.peek()
        if leftover is not None:
            raise QuerySyntaxError(
                f"unbalanced parentheses or trailing input in query {self.query!r}: "
                f"stopped at {leftover.text!r}"
            )
        return expression

    def parse_or(self) -> Expression:
        node = self.parse_and()
        while True:
            token = self.peek()
            if token is None or token.kind != "OR":
                return node
            self.next()
            node = Or(left=node, right=self.parse_and())

    def parse_and(self) -> Expression:
        node = self.parse_unary()
        while True:
            token = self.peek()
            if token is None or token.kind != "AND":
                return node
            self.next()
            node = And(left=node, right=self.parse_unary())

    def parse_unary(self) -> Expression:
        token = self.peek()
        if token is None:
            raise QuerySyntaxError(f"unexpected end of query: {self.query!r}")
        if token.kind == "NOT":
            if not self.allow_not:
                raise QuerySyntaxError(
                    "NOT is only supported in the form 'A AND NOT B'"
                )
            self.next()
            return Not(child=self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expression:
        token = self.next()
        if token.kind == "LPAREN":
            if not self.allow_parentheses:
                raise QuerySyntaxError(f"parentheses are not enabled: {self.query!r}")
            node = self.parse_or()
            self.expect("RPAREN")
            return node
        if token.kind == "RPAREN":
            raise QuerySyntaxError(f"unmatched ')' in query {self.query!r}")
        if token.kind in OPERATORS:
            raise QuerySyntaxError(
                f"operator {token.text} must appear between terms in query {self.query!r}"
            )
        return Term(text=token.text, is_phrase=(token.kind == "PHRASE"))


# ----------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------
def parse_query(
    query: str,
    allow_parentheses: bool = True,
    allow_not: bool = True,
) -> Expression:
    """Parse ``query`` into an AST, raising :class:`QuerySyntaxError` on problems."""
    node = QueryParser(
        query,
        allow_parentheses=allow_parentheses,
        allow_not=allow_not,
    ).parse()
    _check_not_usage(node, query)
    return node


def _check_not_usage(node: Expression, query: str) -> None:
    """Reject ``NOT`` anywhere except directly under an ``AND``.

    ``NOT food`` is not an error to work around: it would return almost every unit
    in the corpus, which tells a user nothing. ``A AND NOT B`` is a real filter,
    so it is the only accepted form.
    """
    if isinstance(node, Term):
        return
    if isinstance(node, Not):
        raise QuerySyntaxError(
            "unrestricted NOT is refused: it would return almost the whole corpus. "
            f"Use the form 'A AND NOT B' instead (query: {query!r})"
        )
    if isinstance(node, Or):
        for child in (node.left, node.right):
            if isinstance(child, Not):
                raise QuerySyntaxError(
                    f"NOT cannot follow OR; write the positive part first, e.g. "
                    f"'(A OR B) AND NOT C' (query: {query!r})"
                )
            _check_not_usage(child, query)
        return
    if isinstance(node, And):
        if isinstance(node.left, Not):
            raise QuerySyntaxError(
                f"write NOT on the right-hand side only, i.e. 'A AND NOT B' (query: {query!r})"
            )
        _check_not_usage(node.left, query)
        if isinstance(node.right, Not):
            # 'A AND NOT B' is the supported form; its operand may not be a bare
            # NOT or an OR, both of which return (almost) the whole corpus.
            if isinstance(node.right.child, (Not, Or)):
                raise QuerySyntaxError(
                    "NOT cannot be applied to another NOT or to an OR; write "
                    f"'A AND NOT B' with a single term (query: {query!r})"
                )
            return
        _check_not_usage(node.right, query)
        return
    raise TypeError(f"unknown AST node {node!r}")


def _contains_not(node: Expression) -> bool:
    if isinstance(node, Not):
        return True
    if isinstance(node, Term):
        return False
    return _contains_not(node.left) or _contains_not(node.right)


def query_type_of(query: str) -> str:
    """Classify a query by the operators it uses (for the result files)."""
    text = (query or "").upper()
    has_and = " AND " in f" {text} "
    has_or = " OR " in f" {text} "
    has_not = " NOT " in f" {text} "
    if "(" in (query or "") and (has_and or has_or or has_not):
        return "boolean_group"
    if has_not:
        # including a bare "NOT x": the parser refuses it, which is the point
        return "boolean_not"
    if has_and:
        return "boolean_and"
    if has_or:
        return "boolean_or"
    if query and query.strip().startswith('"') and query.strip().endswith('"') and len(query.strip()) > 2:
        return "phrase"
    return "keyword"


def describe_expression(node: Expression) -> str:
    """Render an AST back to a readable, re-parsable query string.

    Parentheses are re-inserted wherever precedence requires them, so the printed
    query means exactly what the parsed tree meant.
    """
    return _render(node)


def _render(node: Expression, parent_precedence: int = 0) -> str:
    if isinstance(node, Term):
        return f'"{node.text}"' if node.is_phrase else node.text
    if isinstance(node, Not):
        return f"NOT {_render(node.child, 3)}"
    if isinstance(node, And):
        text = f"{_render(node.left, 2)} AND {_render(node.right, 2)}"
        return f"({text})" if parent_precedence > 2 else text
    if isinstance(node, Or):
        text = f"{_render(node.left, 1)} OR {_render(node.right, 1)}"
        return f"({text})" if parent_precedence > 1 else text
    raise TypeError(f"unknown AST node {node!r}")
