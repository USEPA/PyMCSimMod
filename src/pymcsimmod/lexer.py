from typing import ClassVar

import ply.lex as lex

IDENTIFIER_RE = r"[a-zA-Z_][a-zA-Z_0-9]*"
INTEGER_RE = r"\d+"
FLOAT_RE = r"\d+\.\d*|\.\d+"
EXPONENTIAL_RE = r"(?:\d+\.?\d*|\.\d+)[eE]-?\d+"


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
        # Math functions
        "pow": "POW",
        # Special functions
        "BetaRandom": "BETA_RANDOM",
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
        "NOT_EQUALS",
        "LT",
        "LTE",
        "GT",
        "GTE",
        "QUESTION",
        "COLON",
        "E",
        "ASSIGN",
        *reserved.values(),
    ]

    t_LBRACE = r"\{"
    t_RBRACE = r"\}"
    t_LPAREN = r"\("
    t_RPAREN = r"\)"
    t_COMMA = r","
    t_PLUS = r"\+"
    t_MINUS = r"-"
    t_MULTIPLY = r"\*"
    t_DIVIDE = r"/"
    t_QUESTION = r"\?"
    t_COLON = r":"
    t_ASSIGN = r"="

    t_SEMICOLON = r";"

    t_NOT_EQUALS = r"!="
    t_EQUALS = r"=="
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

    @lex.TOKEN(IDENTIFIER_RE)
    def t_IDENTIFIER(self, t):
        t.type = self.reserved.get(t.value, "IDENTIFIER")  # Check for reserved words
        return t

    @lex.TOKEN(EXPONENTIAL_RE)
    def t_E(self, t):
        t.value = float(t.value)
        return t

    @lex.TOKEN(FLOAT_RE)
    def t_FLOAT(self, t):
        t.value = float(t.value)
        return t

    @lex.TOKEN(INTEGER_RE)
    def t_INTEGER(self, t):
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
