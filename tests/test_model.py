import pytest

from pymcsimmod.model import (
    Condition,
    Identifier,
    MathematicalExpression,
    Number,
    SignedExpression,
    Statement,
)


@pytest.fixture
def number_expr():
    return Number(value=5)


@pytest.fixture
def identifier_expr():
    return Identifier(name="x")


@pytest.fixture
def signed_expr():
    return SignedExpression(sign="-", expression=Number(value=2))


@pytest.fixture
def math_expr():
    # 3 + 4
    return MathematicalExpression(lhs=Number(value=3), operator="+", rhs=Number(value=4))


@pytest.fixture
def math_expr_nested():
    # (2 * 5) - 1
    inner = MathematicalExpression(lhs=Number(value=2), operator="*", rhs=Number(value=5))
    return MathematicalExpression(lhs=inner, operator="-", rhs=Number(value=1))


@pytest.fixture
def condition_expr():
    # 2 < 3
    return Condition(lhs=Number(value=2), operator="<", rhs=Number(value=3))


class TestModel:
    def test_number_and_identifier_eval(self, number_expr, identifier_expr):
        assert number_expr.eval() == 5
        assert identifier_expr.eval() == "x"

    def test_signed_expression_eval(self, signed_expr):
        stmt = Statement(lhs=Identifier(name="y"), rhs=signed_expr)
        d = stmt.to_dict()
        assert d["y"] == -2
