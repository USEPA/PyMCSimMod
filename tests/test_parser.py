from pymcsimmod.model import (
    Condition,
    DtVariable,
    Identifier,
    InitializeSection,
    MathematicalExpression,
    Model,
    Number,
    OutputsSection,
    ParenthesizedExpression,
    SignedExpression,
    Statement,
    StatesSection,
    TernaryExpression,
)
from pymcsimmod.parser import Parser

from .utils import DATA_PATH


def test_empty():
    parser = Parser(start="empty")
    parsed_empty = parser.parse("")
    assert parsed_empty is None


def test_model():
    parser = Parser(start="model")
    parsed_model = parser.parse("States = { x, y, }; Initialize { x = 1.0; } End.")
    expected_model = Model(
        sections=[
            StatesSection(
                type="States",
                declarations=[
                    Identifier(name="x"),
                    Identifier(name="y"),
                ],
            ),
            InitializeSection(
                type="Initialize",
                statements=[Statement(lhs=Identifier(name="x"), rhs=Number(value=1.0))],
            ),
        ]
    )
    assert parsed_model == expected_model


def test_sections():
    parser = Parser(start="sections")
    parsed_sections = parser.parse("States = { x }; Outputs = { y };")
    expected_sections = [
        StatesSection(type="States", declarations=[Identifier(name="x")]),
        OutputsSection(type="Outputs", declarations=[Identifier(name="y")]),
    ]
    assert parsed_sections == expected_sections


def test_section_1():
    parser = Parser(start="section_1")
    parsed_section = parser.parse("States = { x, y };")
    expected_section = StatesSection(
        type="States", declarations=[Identifier(name="x"), Identifier(name="y")]
    )
    assert parsed_section == expected_section


def test_section_2():
    parser = Parser(start="section_2")
    parsed_section = parser.parse("Initialize { x = 1.0; }")
    expected_section = InitializeSection(
        type="Initialize", statements=[Statement(lhs=Identifier(name="x"), rhs=Number(value=1.0))]
    )
    assert parsed_section == expected_section


def test_section_name_1():
    parser = Parser(start="section_name_1")
    parsed_name = parser.parse("States")
    assert parsed_name == "States"


def test_section_name_2():
    parser = Parser(start="section_name_2")
    parsed_name = parser.parse("Initialize")
    assert parsed_name == "Initialize"


def test_section_content_1():
    parser = Parser(start="section_content_1")
    parsed_content = parser.parse("x, y,")
    expected_content = [Identifier(name="x"), Identifier(name="y")]
    assert parsed_content == expected_content


def test_delimited_identifiers():
    parser = Parser(start="delimited_identifiers")
    parsed_identifiers = parser.parse("x, y, z")
    expected_identifiers = [Identifier(name="x"), Identifier(name="y"), Identifier(name="z")]
    assert parsed_identifiers == expected_identifiers


def test_identifier():
    parser = Parser(start="identifier")
    parsed_identifier = parser.parse("x")
    expected_identifier = Identifier(name="x")
    assert parsed_identifier == expected_identifier


def test_statements():
    parser = Parser(start="statements")
    parsed_statements = parser.parse("x = 1.0; y = 2.0;")
    expected_statements = [
        Statement(lhs=Identifier(name="x"), rhs=Number(value=1.0)),
        Statement(lhs=Identifier(name="y"), rhs=Number(value=2.0)),
    ]
    assert parsed_statements == expected_statements


def test_statement():
    parser = Parser(start="statement")
    parsed_statement = parser.parse("a = b * c + (d - e) / f;")
    expected_statement = Statement(
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
    assert parsed_statement == expected_statement


def test_expression():
    parser = Parser(start="expression")
    parsed_expression = parser.parse("b * c")
    expected_expression = MathematicalExpression(
        operator="*", lhs=Identifier(name="b"), rhs=Identifier(name="c")
    )
    assert parsed_expression == expected_expression


def test_variable():
    parser = Parser(start="variable")
    parsed_variable = parser.parse("dt(x)")
    expected_variable = DtVariable(identifier=Identifier(name="x"))
    assert parsed_variable == expected_variable


def test_parenthesized_expression():
    parser = Parser(start="parenthesized_expression")
    parsed_expression = parser.parse("(a + b)")
    expected_expression = ParenthesizedExpression(
        expression=MathematicalExpression(
            operator="+", lhs=Identifier(name="a"), rhs=Identifier(name="b")
        )
    )
    assert parsed_expression == expected_expression


def test_ternary_expression():
    parser = Parser(start="ternary_expression")
    parsed_expression = parser.parse("a < b ? c : d")
    expected_expression = TernaryExpression(
        condition=Condition(operator="<", lhs=Identifier(name="a"), rhs=Identifier(name="b")),
        if_true=Identifier(name="c"),
        if_false=Identifier(name="d"),
    )
    assert parsed_expression == expected_expression


def test_condition():
    parser = Parser(start="condition")
    parsed_condition = parser.parse("a < b")
    expected_condition = Condition(operator="<", lhs=Identifier(name="a"), rhs=Identifier(name="b"))
    assert parsed_condition == expected_condition


def test_mathematical_expression():
    parser = Parser(start="mathematical_expression")
    parsed_expression = parser.parse("a + b")
    expected_expression = MathematicalExpression(
        operator="+", lhs=Identifier(name="a"), rhs=Identifier(name="b")
    )
    assert parsed_expression == expected_expression


def test_number():
    parser = Parser(start="number")
    parsed_number = parser.parse("123.45")
    expected_number = Number(value=123.45)
    assert parsed_number == expected_number


def test_signed_expression():
    parser = Parser(start="signed_expression")
    parsed_expression = parser.parse("-a")
    expected_expression = SignedExpression(sign="-", expression=Identifier(name="a"))
    assert parsed_expression == expected_expression


def test_to_mod():
    model = (DATA_PATH / "good.model").read_text()
    parser = Parser()
    parsed_model = parser.parse(model)
    assert parsed_model.to_mod() == model
