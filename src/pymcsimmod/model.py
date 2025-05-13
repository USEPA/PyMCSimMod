from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# Variables / Constants


class Identifier(BaseModel):
    name: str

    def to_mod(self) -> str:
        return self.name
    def eval(self) -> str:
        return self.name


class DtVariable(BaseModel):
    identifier: Identifier

    def to_mod(self) -> str:
        return f"dt({self.identifier.to_mod()})"


Variable = Identifier | DtVariable


class Number(BaseModel):
    value: int | float

    def to_mod(self) -> str:
        return str(self.value)
    def eval(self):
        return self.value



# Mathematical functions


class PowFunction(BaseModel):
    func: Literal["pow"] = "pow"
    args: list[Expression]

    def to_mod(self) -> str:
        return f"""pow({", ".join(arg.to_mod() for arg in self.args)})"""


MathematicalFunction = PowFunction


# Special functions


class BetaRandomFunction(BaseModel):
    func: Literal["BetaRandom"] = "BetaRandom"
    args: list[Expression]

    def to_mod(self) -> str:
        return f"""BetaRandom({", ".join(arg.to_mod() for arg in self.args)})"""


SpecialFunction = BetaRandomFunction


# Expressions


class ParenthesizedExpression(BaseModel):
    expression: Expression

    def to_mod(self) -> str:
        return f"({self.expression.to_mod()})"


class MathematicalExpression(BaseModel):
    operator: str
    lhs: Expression
    rhs: Expression

    def to_mod(self) -> str:
        return f"{self.lhs.to_mod()} {self.operator} {self.rhs.to_mod()}"

    def eval(self, vars: dict[float | int]):
        lhs_value = self.lhs.eval(vars)
        rhs_value = self.rhs.eval(vars)
        if self.operator == '+':
            return lhs_value + rhs_value
        elif self.operator == '-':
            return lhs_value - rhs_value
        elif self.operator == '*':
            return lhs_value * rhs_value
        elif self.operator == '/':
            return lhs_value / rhs_value
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

class SignedExpression(BaseModel):
    sign: str
    expression: Variable | Number | ParenthesizedExpression

    def to_mod(self) -> str:
        return f"{self.sign}{self.expression.to_mod()}"


class Condition(BaseModel):
    operator: str
    lhs: Expression
    rhs: Expression

    def to_mod(self) -> str:
        return f"{self.lhs.to_mod()} {self.operator} {self.rhs.to_mod()}"


class TernaryExpression(BaseModel):
    condition: Condition
    if_true: Expression
    if_false: Expression

    def to_mod(self) -> str:
        return f"{self.condition.to_mod()} ? {self.if_true.to_mod()} : {self.if_false.to_mod()}"


Expression = (
    SignedExpression
    | MathematicalFunction
    | SpecialFunction
    | MathematicalExpression
    | TernaryExpression
    | Variable
    | Number
    | ParenthesizedExpression
)

# Sections


class Statement(BaseModel):
    lhs: Variable
    rhs: Expression

    def to_mod(self) -> str:
        return f"{self.lhs.to_mod()} = {self.rhs.to_mod()};"
    def to_dict(self) -> dict[str, int | float]:
        return {self.lhs.eval(): self.rhs.eval()}
    
    @property
    def is_constant(self) -> bool:
        """
        Check if the statement is a constant assignment.
        A statement is considered a constant assignment if the right-hand side is a number.
        """
        return isinstance(self.rhs, Number) # TODO: Or is signed expression(Number)


class StatesSection(BaseModel):
    type: Literal["States"] = "States"
    declarations: list[Variable]

    def to_mod(self) -> str:
        return f"""States = {{\n{",\n".join(f"    {declaration.to_mod()}" for declaration in self.declarations)}\n}};"""


class InputsSection(BaseModel):
    type: Literal["Inputs"] = "Inputs"
    declarations: list[Variable]

    def to_mod(self) -> str:
        return f"""Inputs = {{\n{",\n".join(f"    {declaration.to_mod()}" for declaration in self.declarations)}\n}};"""


class OutputsSection(BaseModel):
    type: Literal["Outputs"] = "Outputs"
    declarations: list[Variable]

    def to_mod(self) -> str:
        return f"""Outputs = {{\n{",\n".join(f"    {declaration.to_mod()}" for declaration in self.declarations)}\n}};"""


class InitializeSection(BaseModel):
    type: Literal["Initialize"] = "Initialize"
    statements: list[Statement]

    def to_mod(self) -> str:
        return f"""Initialize {{\n{"\n".join(f"    {statement.to_mod()}" for statement in self.statements)}\n}}"""


class DynamicsSection(BaseModel):
    type: Literal["Dynamics"] = "Dynamics"
    statements: list[Statement]

    def to_mod(self) -> str:
        return f"""Dynamics {{\n{"\n".join(f"    {statement.to_mod()}" for statement in self.statements)}\n}}"""


class JacobianSection(BaseModel):
    type: Literal["Jacobian"] = "Jacobian"
    statements: list[Statement]

    def to_mod(self) -> str:
        return f"""Jacobian {{\n{"\n".join(f"    {statement.to_mod()}" for statement in self.statements)}\n}}"""


class CalcOutputsSection(BaseModel):
    type: Literal["CalcOutputs"] = "CalcOutputs"
    statements: list[Statement]

    def to_mod(self) -> str:
        return f"""CalcOutputs {{\n{"\n".join(f"    {statement.to_mod()}" for statement in self.statements)}\n}}"""


Section = Annotated[
    StatesSection
    | InputsSection
    | OutputsSection
    | InitializeSection
    | DynamicsSection
    | JacobianSection
    | CalcOutputsSection,
    Field(discriminator="type"),
]


class Model(BaseModel):
    sections: list[Section | Statement]

    def to_mod(self) -> str:
        return f"""{"\n".join(section.to_mod() for section in self.sections)}\nEnd."""

    @property
    def parameters(self) -> dict[str, float | int]:
        """
        Extract parameters from the model.
        Returns a dictionary where keys are parameter names and values are their numeric values.
        """
        params = {}
        for section in self.sections:
            if isinstance(section, Statement):
                params.update(section.to_dict())
        return params
    @property
    def Y0(self) -> dict[str, float | int]:
        """
        Extract initial conditions from the model.
        Returns a dictionary where keys are variable names and values are their initial values.
        """
        Y0 = {}
        for section in self.sections:
            if isinstance(section, InitializeSection):
                for statement in section.statements:
                    Y0.update(statement.to_dict())
        return Y0
    @property
    def dynamics(self) -> dict[str | int]:
        """
        Extract only the dydt dynamics from the model.
        Returns a dictionary where keys are variable names and values are their dynamics.
        """
        dynamics = {}
        for section in self.sections:
            if isinstance(section, DynamicsSection):
                for statement in section.statements:
                    if isinstance(statement.lhs, DtVariable):
                        variable = statement.lhs.identifier.name
                        dynamics[variable] = statement.rhs
        return dynamics
    @property
    def dynamic_calcs(self) -> dict[str | int]:
        """
        Extract any calculations from the dynamics section from the model.
        Returns a dictionary where keys are variable names and values are their dynamics calculations.
        """
        dynamics_calcs = {}
        for section in self.sections:
            if isinstance(section, DynamicsSection):
                for statement in section.statements:
                    if isinstance(statement.lhs, Identifier):
                        variable = statement.lhs.name
                        dynamics_calcs[variable] = statement.rhs
        return dynamics_calcs

    def evaluate_expression(self, expr: MathematicalExpression, context: dict[str, float | int]) -> float | int:
        """
        Recursively evaluate an expression using the provided context dict.
        Context should include state variables, parameters, and any calculated variables.
        """
        # Identifier
        if isinstance(expr, Identifier):
            name = expr.name
            if name in context:
                return context[name]
            raise KeyError(f"Unknown identifier '{name}' in expression.")
        # Number
        elif isinstance(expr, Number):
            return expr.value
        # SignedExpression (must be checked before generic hasattr checks)
        elif isinstance(expr, SignedExpression):
            val = self.evaluate_expression(expr.expression, context)
            return val if expr.sign == '+' else -val
        # MathematicalExpression
        elif isinstance(expr, MathematicalExpression):
            lhs = self.evaluate_expression(expr.lhs, context)
            rhs = self.evaluate_expression(expr.rhs, context)
            if expr.operator == '+':
                return lhs + rhs
            elif expr.operator == '-':
                return lhs - rhs
            elif expr.operator == '*':
                return lhs * rhs
            elif expr.operator == '/':
                return lhs / rhs
            else:
                raise ValueError(f"Unknown operator '{expr.operator}' in expression.")
        # ParenthesizedExpression
        elif isinstance(expr, ParenthesizedExpression):
            return self.evaluate_expression(expr.expression, context)
        # TernaryExpression
        elif hasattr(expr, 'condition') and hasattr(expr, 'if_true') and hasattr(expr, 'if_false'):
            cond = expr.condition
            lhs = self.evaluate_expression(cond.lhs, context)
            rhs = self.evaluate_expression(cond.rhs, context)
            op = cond.operator
            if op == '==':
                result = lhs == rhs
            elif op == '!=':
                result = lhs != rhs
            elif op == '<':
                result = lhs < rhs
            elif op == '>':
                result = lhs > rhs
            elif op == '<=':
                result = lhs <= rhs
            elif op == '>=':
                result = lhs >= rhs
            else:
                raise ValueError(f"Unknown condition operator: {op}")
            return self.evaluate_expression(expr.if_true, context) if result else self.evaluate_expression(expr.if_false, context)
        # MathematicalFunction (e.g., pow)
        elif hasattr(expr, 'func') and hasattr(expr, 'args'):
            if expr.func == 'pow':
                args = [self.evaluate_expression(arg, context) for arg in expr.args]
                return pow(*args)
            else:
                raise ValueError(f"Unknown function: {expr.func}")
        # SpecialFunction (e.g., BetaRandom)
        elif hasattr(expr, 'func') and hasattr(expr, 'args'):
            # For now, just return 0 or raise (implement as needed)
            raise NotImplementedError(f"Special function {expr.func} not implemented.")
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")