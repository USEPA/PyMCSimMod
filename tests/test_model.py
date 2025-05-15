import operator

import pytest

from pymcsimmod.model import (
    Condition,
    Identifier,
    MathematicalExpression,
    Number,
    SignedExpression,
    Statement,
)

expression_map = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "pow": pow,
}

condition_map = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
}


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
def condition_expr():
    # 2 < 3
    return Condition(lhs=Number(value=2), operator="<", rhs=Number(value=3))


class TestModel:
    def test_number_eval(self, number_expr):
        assert number_expr.eval() == 5

    def test_signed_expression_eval(self, signed_expr):
        assert signed_expr.eval() == -2

    def test_identifier_fixture(self, identifier_expr):
        # Identifier does not have .eval(), but we can check its name
        assert identifier_expr.name == "x"

    def test_math_expr_fixture(self, math_expr):
        # MathematicalExpression does not have .eval(), but we can check its structure
        assert math_expr.operator == "+"
        assert isinstance(math_expr.lhs, Number)
        assert isinstance(math_expr.rhs, Number)
        assert expression_map[math_expr.operator](math_expr.lhs.eval(), math_expr.rhs.eval()) == 7

    def test_condition_fixture(self, condition_expr):
        # Condition does not have .eval(), but we can check its structure
        assert condition_expr.operator == "<"
        assert isinstance(condition_expr.lhs, Number)
        assert isinstance(condition_expr.rhs, Number)
        assert condition_map[condition_expr.operator](
            condition_expr.lhs.eval(), condition_expr.rhs.eval()
        )

    def test_statement_to_dict_with_signed(self, signed_expr):
        stmt = Statement(lhs=Identifier(name="y"), rhs=signed_expr)
        d = stmt.to_dict()
        assert d["y"] == -2
