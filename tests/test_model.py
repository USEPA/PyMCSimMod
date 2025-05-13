import pytest
from pymcsimmod.model import Identifier, Number, SignedExpression, Statement

@pytest.fixture
def number_expr():
    return Number(value=5)

@pytest.fixture
def identifier_expr():
    return Identifier(name="x")

@pytest.fixture
def signed_expr():
    return SignedExpression(sign="-", expression=Number(value=2))

class TestModel:
    def test_number_and_identifier_eval(self, number_expr, identifier_expr):
        assert number_expr.eval() == 5
        assert identifier_expr.eval() == "x"

    def test_signed_expression_eval(self, signed_expr):
        stmt = Statement(lhs=Identifier(name="y"), rhs=signed_expr)
        d = stmt.to_dict()
        assert d["y"] == -2
