from pymcsimmod.model import Identifier, MathematicalExpression, ParenthesizedExpression, Statement
from pymcsimmod.parser import Parser


def test_statement():
    parser = Parser(start="statement")
    parsed_statement = parser.parse("a = b * c + (d - e) / f;")
    assert parsed_statement == Statement(
        lhs=Identifier(name="a"),
        rhs=MathematicalExpression(
            operator="+",
            lhs=MathematicalExpression(
                operator="*", lhs=Identifier(name="b"), rhs=Identifier(name="c")
            ),
            rhs=MathematicalExpression(
                operator="/",
                lhs=ParenthesizedExpression(
                    expression=MathematicalExpression(
                        operator="-", lhs=Identifier(name="d"), rhs=Identifier(name="e")
                    )
                ),
                rhs=Identifier(name="f"),
            ),
        ),
    )
