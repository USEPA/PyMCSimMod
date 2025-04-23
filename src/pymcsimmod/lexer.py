from typing import ClassVar

import ply.lex as lex


class Lexer:
    reserved: ClassVar = {
        # Sections
        "States": "STATES",
        "Inputs": "INPUTS",
        "Outputs": "OUTPUTS",
        "Initialize": "INITIALIZE",
        "Dynamics": "DYNAMICS",
        "Jacobian": "JACOBIAN",
        "CalcOutputs": "CALC_OUTPUTS",
        # Reserved keywords
        "dt": "DT",
        # C math
        "pow": "POW",
    }

    tokens: ClassVar = [
        "IDENTIFIER",
        "LBRACE",
        "RBRACE",
        "LPAREN",
        "RPAREN",
        "SEMICOLON",
        "FLOAT",
        "INTEGER",
        "END",
        "COMMA",
        "PLUS",
        "MINUS",
        "MULTIPLY",
        "DIVIDE",
        "EQUALS",
        "LT",
        "LTE",
        "GT",
        "GTE",
        "QUESTION",
        "COLON",
        "E",
        *reserved.values(),
    ]

    t_LBRACE = r"\{"
    t_RBRACE = r"\}"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_COMMA = r"\,"
    t_PLUS = r"\+"
    t_MINUS = r"\-"
    t_MULTIPLY = r"\*"
    t_DIVIDE = r"\/"
    t_EQUALS = r"\="
    t_QUESTION = r"\?"
    t_COLON = r"\:"

    t_SEMICOLON = r"\;"

    t_LT = r"<"
    t_LTE = r"<="
    t_GT = r">"
    t_GTE = r">="

    def t_COMMENT(self, t):
        r"\#.*"
        pass

    def t_END(self, t):
        r"End\."
        return t

    def t_IDENTIFIER(self, t):
        r"[a-zA-Z_][a-zA-Z_0-9]*"
        t.type = self.reserved.get(t.value, "IDENTIFIER")  # Check for reserved words
        return t

    def t_E(self, t):
        r"\-?((\d+\.\d*)|(\d*\.\d+))[eE]\-?\d+"
        t.value = float(t.value)
        return t

    def t_FLOAT(self, t):
        r"\-?((\d+\.\d*)|(\d*\.\d+))"
        t.value = float(t.value)
        return t

    def t_INTEGER(self, t):
        r"\d+"
        t.value = int(t.value)
        return t

    def t_newline(self, t):
        r"\n+"
        t.lexer.lineno += len(t.value)

    t_ignore = " \t"

    def __init__(self):
        self.build()

    def build(self, **kwargs):
        self.lexer: lex.Lexer = lex.lex(module=self, **kwargs)
