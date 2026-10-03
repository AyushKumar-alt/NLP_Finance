"""The Boolean query parser: accepted forms, rejected forms, no eval()."""

from __future__ import annotations

import ast
import inspect

import pytest

from src.phase3 import query_parser
from src.phase3.query_parser import (
    And,
    Not,
    Or,
    QuerySyntaxError,
    Term,
    describe_expression,
    parse_query,
    query_type_of,
    tokenize_query,
)


# ----------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------
def test_tokenizer_splits_operators_terms_and_phrases():
    tokens = tokenize_query('rbi AND "monetary policy" OR (gdp)')
    assert [(token.kind, token.text) for token in tokens] == [
        ("TERM", "rbi"),
        ("AND", "AND"),
        ("PHRASE", "monetary policy"),
        ("OR", "OR"),
        ("LPAREN", "("),
        ("TERM", "gdp"),
        ("RPAREN", ")"),
    ]


def test_tokenizer_rejects_an_empty_quoted_phrase():
    with pytest.raises(QuerySyntaxError):
        tokenize_query('a AND ""')


def test_tokenizer_rejects_an_empty_query():
    with pytest.raises(QuerySyntaxError):
        tokenize_query("   ")


# ----------------------------------------------------------------------
# accepted grammar
# ----------------------------------------------------------------------
def test_single_term():
    node = parse_query("inflation")
    assert node == Term(text="inflation", is_phrase=False)


def test_quoted_phrase_is_flagged():
    node = parse_query('"monetary policy"')
    assert node == Term(text="monetary policy", is_phrase=True)


def test_and_has_lower_precedence_than_or():
    node = parse_query("a AND b OR c")
    assert node == Or(left=And(left=Term("a"), right=Term("b")), right=Term("c"))


def test_or_has_lower_precedence_than_and():
    node = parse_query("a OR b AND c")
    assert node == Or(left=Term("a"), right=And(left=Term("b"), right=Term("c")))


def test_not_binds_tighter_than_and():
    node = parse_query("a AND NOT b")
    assert node == And(left=Term("a"), right=Not(child=Term("b")))


def test_parentheses_override_precedence():
    node = parse_query("(a OR b) AND c")
    assert node == And(left=Or(left=Term("a"), right=Term("b")), right=Term("c"))


def test_left_associativity():
    node = parse_query("a AND b AND c")
    assert node == And(left=And(left=Term("a"), right=Term("b")), right=Term("c"))


def test_query_may_mix_operators_and_phrases():
    node = parse_query('(gdp OR "gross domestic product") AND NOT food')
    assert isinstance(node, And)
    assert isinstance(node.left, Or)
    assert isinstance(node.right, Not)
    assert node.left.left.is_phrase is False
    assert node.left.right.is_phrase is True


def test_nested_parentheses():
    node = parse_query("((a OR b) AND (c AND NOT d))")
    assert isinstance(node, And)
    assert isinstance(node.left, Or)
    assert isinstance(node.right, And)


def test_keywords_are_case_insensitive():
    assert parse_query("gdp and inflation") == parse_query("gdp AND inflation")


# ----------------------------------------------------------------------
# rejected grammar
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    [
        "NOT food",                 # unrestricted NOT is refused
        "NOT (food OR banking)",
        "NOT food AND gdp",
        "food AND (",               # unbalanced
        "food)",
        "(food",
        "()",
        "AND food",                 # operator without a left operand
        "food AND",
        "food OR",
        '"unclosed phrase',
        "food AND AND gdp",
        "(a OR NOT b) AND c",
    ],
)
def test_malformed_queries_raise(query):
    with pytest.raises(QuerySyntaxError):
        parse_query(query)


def test_bare_not_error_message_names_the_supported_form():
    with pytest.raises(QuerySyntaxError) as error:
        parse_query("NOT food")
    assert "A AND NOT B" in str(error.value)


def test_not_can_be_disabled_entirely():
    with pytest.raises(QuerySyntaxError):
        parse_query("a AND NOT b", allow_not=False)
    assert parse_query("a AND b", allow_not=False) is not None


def test_parentheses_can_be_disabled():
    with pytest.raises(QuerySyntaxError):
        parse_query("(a OR b) AND c", allow_parentheses=False)


def test_or_under_not_is_always_refused():
    with pytest.raises(QuerySyntaxError):
        parse_query("a AND NOT (b OR c)")
    assert parse_query("(a OR b) AND NOT c") is not None


def test_parser_never_uses_eval_or_exec():
    """The security claim is checked in the source, not just asserted."""
    source = inspect.getsource(query_parser)
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "eval" not in called
    assert "exec" not in called
    assert "compile" not in called


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,expected",
    [
        ("inflation", "keyword"),
        ("GDP", "keyword"),
        ('"monetary policy"', "phrase"),
        ("gdp AND inflation", "boolean_and"),
        ("gdp OR gva", "boolean_or"),
        ("inflation AND NOT food", "boolean_not"),
        ("(GDP OR GVA) AND policy", "boolean_group"),
    ],
)
def test_query_type_detection(query, expected):
    assert query_type_of(query) == expected


def test_describe_expression_round_trips_a_readable_query():
    node = parse_query('(gdp OR gva) AND NOT food')
    assert describe_expression(node) == '(gdp OR gva) AND NOT food'
