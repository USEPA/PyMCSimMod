import ply.yacc as yacc
from pydantic import TypeAdapter

from .lexer import Lexer
from .model import (
    Condition,
    DtVariable,
    Identifier,
    MathematicalExpression,
    Model,
    NegativeExpression,
    Number,
    ParenthesizedExpression,
    PowFunction,
    Section,
    Statement,
    TernaryExpression,
)


class Parser:
    tokens = Lexer.tokens
    start = "model"

    precedence = (
        ("right", "QUESTION", "COLON"),
        ("left", "PLUS", "MINUS"),
        ("left", "MULTIPLY", "DIVIDE"),
    )

    def p_empty(self, p):
        "empty :"
        p[0] = None

    def p_model(self, p):
        "model : sections END"
        p[0] = Model(sections=p[1])

    def p_sections(self, p):
        """sections : sections section
        | section"""
        p[0] = [*p[1], p[2]] if len(p) == 3 else [p[1]]

    def p_section(self, p):
        """section : section_1
        | section_2
        | statement"""
        p[0] = p[1]

    def p_section_1(self, p):
        "section_1 : section_name_1 EQUALS LBRACE section_content_1 RBRACE SEMICOLON"
        p[0] = TypeAdapter(Section).validate_python({"type": p[1], "declarations": p[4]})

    def p_section_2(self, p):
        "section_2 : section_name_2 LBRACE statements RBRACE"
        p[0] = TypeAdapter(Section).validate_python({"type": p[1], "statements": p[3]})

    def p_section_name(self, p):
        """section_name_1 : STATES
                       | INPUTS
                       | OUTPUTS
        section_name_2 : INITIALIZE
                       | DYNAMICS
                       | JACOBIAN
                       | CALC_OUTPUTS"""
        p[0] = p[1]

    def p_section_content_1(self, p):
        """section_content_1 : delimited_identifiers
        | delimited_identifiers COMMA
        | empty"""
        p[0] = p[1] if p[1] else []

    def p_delimited_identifiers(self, p):
        """delimited_identifiers : delimited_identifiers COMMA identifier
        | identifier"""
        p[0] = [*p[1], p[3]] if len(p) == 4 else [p[1]]

    def p_identifier(self, p):
        "identifier : IDENTIFIER"
        p[0] = Identifier(name=p[1])

    def p_statements(self, p):
        """statements : statements statement
        | empty"""
        p[0] = [*p[1], p[2]] if len(p) == 3 else ([p[1]] if p[1] else [])

    def p_statement(self, p):
        "statement : variable EQUALS expression SEMICOLON"
        p[0] = Statement(lhs=p[1], rhs=p[3])

    def p_expression(self, p):
        """expression : parenthesized_expression
        | ternary_expression
        | mathematical_expression
        | negative_expression
        | number
        | variable"""
        p[0] = p[1]

    def p_variable(self, p):
        """variable : DT LPAREN identifier RPAREN
        | identifier"""
        p[0] = DtVariable(identifier=p[3]) if p[1] == "dt" else p[1]

    def p_parenthesized_expression(self, p):
        "parenthesized_expression : LPAREN expression RPAREN"
        p[0] = ParenthesizedExpression(expression=p[2])

    def p_ternary_expression(self, p):
        "ternary_expression : condition QUESTION expression COLON expression"
        p[0] = TernaryExpression(condition=p[1], if_true=p[3], if_false=p[5])

    def p_equality_operator(self, p):
        "equality_operator : EQUALS EQUALS"
        p[0] = "=="

    def p_condition_operator(self, p):
        """condition_operator : equality_operator
        | LT
        | LTE
        | GT
        | GTE"""
        p[0] = p[1]

    def p_condition(self, p):
        "condition : expression condition_operator expression"
        p[0] = Condition(operator=p[2], lhs=p[1], rhs=p[3])

    def p_mathematical_expression(self, p):
        """mathematical_expression : expression PLUS expression
        | expression MINUS expression
        | expression MULTIPLY expression
        | expression DIVIDE expression"""
        p[0] = MathematicalExpression(operator=p[2], lhs=p[1], rhs=p[3])

    def p_mathematical_expression_2(self, p):
        "mathematical_expression : POW LPAREN expression COMMA expression RPAREN"
        p[0] = PowFunction(func=p[1], args=[p[3], p[5]])

    def p_number(self, p):
        """number : INTEGER
        | FLOAT
        | E"""
        p[0] = Number(value=p[1])

    def p_negative_expression(self, p):
        """negative_expression : MINUS number
        | MINUS variable
        | MINUS parenthesized_expression"""
        p[0] = NegativeExpression(expression=p[2])

    def __init__(self, start=None):
        if start is not None:
            self.start = start
        self.lexer = Lexer()
        self.build()

    def build(self, **kwargs):
        kwargs.setdefault("debug", False)
        self.parser: yacc.LRParser = yacc.yacc(module=self, **kwargs)

    def parse(self, data, **kwargs):
        kwargs.setdefault("lexer", self.lexer.lexer)
        return self.parser.parse(data, **kwargs)
