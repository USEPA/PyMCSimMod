from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Literal

import jax
from pydantic import BaseModel, Field
from scipy import stats


class Approach(Enum):
    JAX = 1
    SCIPY = 2


# Variables / Constants


class Identifier(BaseModel):
    name: str

    def evaluate(self, context, approach):
        return context[self.name]

    def to_mod(self) -> str:
        return self.name

    def eval(self) -> str:
        return self.name


class DtVariable(BaseModel):
    identifier: Identifier

    @property
    def name(self):
        return self.identifier.name

    def evaluate(self, context, approach):
        return context[self.name]

    def to_mod(self) -> str:
        return f"dt({self.identifier.to_mod()})"


Variable = Identifier | DtVariable


class Number(BaseModel):
    value: int | float

    def evaluate(self, context, approach):
        return self.value

    def to_mod(self) -> str:
        return str(self.value)

    def eval(self):
        return self.value


# Mathematical functions


class MathematicalFunction(BaseModel):
    func: Literal["log", "log10", "sqrt", "pow", "exp"]
    args: list[Expression]

    def evaluate(self, context, approach):
        args = [arg.evaluate(context, approach) for arg in self.args]
        if self.func == "log":
            return math.log(*args)
        if self.func == "log10":
            return math.log10(*args)
        if self.func == "sqrt":
            return math.sqrt(*args)
        if self.func == "pow":
            return math.pow(*args)
        if self.func == "exp":
            return math.exp(*args)

    def to_mod(self) -> str:
        return f"""{self.func}({", ".join(arg.to_mod() for arg in self.args)})"""


# Special functions


class SpecialFunction(BaseModel):
    func: Literal["BetaRandom"]
    args: list[Expression]

    def evaluate(self, context, approach):
        args = [arg.evaluate(context, approach) for arg in self.args]
        if self.func == "BetaRandom":
            alpha, beta, a, b = args
            if approach == Approach.JAX:
                key = jax.random.PRNGKey(0)
                return jax.random.beta(key, a, b, shape=(alpha, beta))
            if approach == Approach.SCIPY:
                return stats.beta.rvs(a, b, size=(alpha, beta))

    def to_mod(self) -> str:
        return f"""{self.func}({", ".join(arg.to_mod() for arg in self.args)})"""


# Expressions


class ParenthesizedExpression(BaseModel):
    expression: Expression

    def evaluate(self, context, approach):
        return self.expression.evaluate(context, approach)

    def to_mod(self) -> str:
        return f"({self.expression.to_mod()})"


class MathematicalExpression(BaseModel):
    operator: Literal["+", "-", "*", "/"]
    lhs: Expression
    rhs: Expression

    def evaluate(self, context, approach):
        if self.operator == "+":
            return self.lhs.evaluate(context, approach) + self.rhs.evaluate(context, approach)
        if self.operator == "-":
            return self.lhs.evaluate(context, approach) - self.rhs.evaluate(context, approach)
        if self.operator == "*":
            return self.lhs.evaluate(context, approach) * self.rhs.evaluate(context, approach)
        if self.operator == "/":
            return self.lhs.evaluate(context, approach) / self.rhs.evaluate(context, approach)

    def to_mod(self) -> str:
        return f"{self.lhs.to_mod()} {self.operator} {self.rhs.to_mod()}"


class SignedExpression(BaseModel):
    sign: Literal["+", "-"]
    expression: Variable | Number | ParenthesizedExpression

    def evaluate(self, context, approach):
        if self.sign == "+":
            return self.expression.evaluate(context, approach)
        if self.sign == "-":
            return -self.expression.evaluate(context, approach)

    def to_mod(self) -> str:
        return f"{self.sign}{self.expression.to_mod()}"

    def eval(self) -> float | int:
        val = self.expression.eval()
        return val if self.sign == "+" else -val


class Condition(BaseModel):
    operator: Literal["==", "!=", "<", ">", "<=", ">="]
    lhs: Expression
    rhs: Expression

    def evaluate(self, context, approach):
        if self.operator == "==":
            return self.lhs.evaluate(context, approach) == self.rhs.evaluate(context, approach)
        if self.operator == "!=":
            return self.lhs.evaluate(context, approach) != self.rhs.evaluate(context, approach)
        if self.operator == "<":
            return self.lhs.evaluate(context, approach) < self.rhs.evaluate(context, approach)
        if self.operator == ">":
            return self.lhs.evaluate(context, approach) > self.rhs.evaluate(context, approach)
        if self.operator == "<=":
            return self.lhs.evaluate(context, approach) <= self.rhs.evaluate(context, approach)
        if self.operator == ">=":
            return self.lhs.evaluate(context, approach) >= self.rhs.evaluate(context, approach)

    def to_mod(self) -> str:
        return f"{self.lhs.to_mod()} {self.operator} {self.rhs.to_mod()}"


class TernaryExpression(BaseModel):
    condition: Condition
    if_true: Expression
    if_false: Expression

    def evaluate(self, context, approach):
        return (
            self.if_true.evaluate(context, approach)
            if self.condition.evaluate(context, approach)
            else self.if_false.evaluate(context, approach)
        )

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

    @property
    def dynamic(self):
        return isinstance(self.lhs, DtVariable)

    def to_mod(self) -> str:
        return f"{self.lhs.to_mod()} = {self.rhs.to_mod()};"

    def to_dict(self) -> dict[str, int | float]:
        # Evaluate the right-hand side, handling SignedExpression recursively
        def eval_rhs(expr):
            if hasattr(expr, "eval"):  # Handles Number and SignedExpression
                return expr.eval()
            elif hasattr(expr, "value"):
                return expr.value
            else:
                raise TypeError(f"Unsupported expression type in Statement.to_dict: {type(expr)}")

        return {self.lhs.eval(): eval_rhs(self.rhs)}


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

    def get_Y0(self, context, approach):
        """
        Evaluate initial conditions using the given context and approach.

        Args:
            context: Dictionary of parameter values for evaluation.
            approach: Approach enum (JAX or SCIPY) for evaluation.

        Returns:
            Dictionary of evaluated initial conditions.
        """
        Y0 = {}
        for statement in self.statements:
            var_name = (
                statement.lhs.eval() if hasattr(statement.lhs, "eval") else statement.lhs.name
            )
            Y0[var_name] = statement.rhs.evaluate(context, approach)
        return Y0

    def to_mod(self) -> str:
        return f"""Initialize {{\n{"\n".join(f"    {statement.to_mod()}" for statement in self.statements)}\n}}"""


class DynamicsSection(BaseModel):
    type: Literal["Dynamics"] = "Dynamics"
    statements: list[Statement]

    def get_dynamics(self, context, approach):
        dynamics = []
        for statement in self.statements:
            if statement.dynamic:
                dynamics.append(statement.rhs.evaluate(context, approach))
            else:
                context[statement.lhs.name] = statement.rhs.evaluate(context, approach)
        return dynamics

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

    @property
    def dynamics_section(self):
        return next((_ for _ in self.sections if isinstance(_, DynamicsSection)), None)

    def to_mod(self) -> str:
        return f"""{"\n".join(section.to_mod() for section in self.sections)}\nEnd."""

    @property
    def parameters(self) -> dict[str, float | int]:
        """
        Extract parameters from the model.
        Returns a dictionary where keys are parameter names and values are their numeric values.
        """
        if not hasattr(self, "_params"):
            params = {}
            for section in self.sections:
                if isinstance(section, Statement):
                    params.update(section.to_dict())
            self._params = params
        return self._params

    @property
    def inputs(self) -> list[str]:
        """
        Extract inputs from the model.
        Returns a list of input names.
        """
        if not hasattr(self, "_inputs"):
            self._inputs = []  # Default to empty list
            for section in self.sections:
                if isinstance(section, InputsSection):
                    self._inputs = [declaration.name for declaration in section.declarations]
                    break
        return self._inputs

    @property
    def outputs(self) -> list[str]:
        """
        Extract outputs from the model.
        Returns a list of output names.
        """
        if not hasattr(self, "_outputs"):
            self._outputs = []  # Default to empty list
            for section in self.sections:
                if isinstance(section, OutputsSection):
                    self._outputs = [declaration.name for declaration in section.declarations]
                    break
        return self._outputs

    @property
    def Y0(self) -> dict[str, float | int]:
        """
        Extract initial conditions from the model.
        Returns a dictionary where keys are variable names and values are their initial values.
        """
        if not hasattr(self, "_Y0"):
            Y0 = {}
            for section in self.sections:
                if isinstance(section, InitializeSection):
                    for statement in section.statements:
                        Y0.update(statement.to_dict())
            self._Y0 = Y0
        return self._Y0

    @property
    def dynamics(self) -> dict[str, MathematicalExpression]:
        """
        Extract only the dydt dynamics from the model.
        Returns a dictionary where keys are variable names and values are their dynamics.
        """
        if not hasattr(self, "_dynamics"):
            dynamics = {}
            for section in self.sections:
                if isinstance(section, DynamicsSection):
                    for statement in section.statements:
                        if isinstance(statement.lhs, DtVariable):
                            variable = statement.lhs.identifier.name
                            dynamics[variable] = statement.rhs
            self._dynamics = dynamics
        return self._dynamics

    @property
    def dynamic_calcs(self) -> dict[str, MathematicalExpression]:
        """
        Extract any calculations from the dynamics section from the model.
        Returns a dictionary where keys are variable names and values are their dynamics calculations.
        """
        if not hasattr(self, "_dynamics_calcs"):
            dynamics_calcs = {}
            for section in self.sections:
                if isinstance(section, DynamicsSection):
                    for statement in section.statements:
                        if isinstance(statement.lhs, Identifier):
                            variable = statement.lhs.name
                            dynamics_calcs[variable] = statement.rhs
            self._dynamics_calcs = dynamics_calcs
        return self._dynamics_calcs

    @property
    def calc_outputs(self) -> dict[str, MathematicalExpression]:
        """
        Extract any calculations from the dynamics section from the model.
        Returns a dictionary where keys are variable names and values are their dynamics calculations.
        """
        if not hasattr(self, "_calc_outputs"):
            calc_outputs = {}
            for section in self.sections:
                if isinstance(section, CalcOutputsSection):
                    for statement in section.statements:
                        if isinstance(statement.lhs, Identifier):
                            variable = statement.lhs.name
                            calc_outputs[variable] = statement.rhs
            self._calc_outputs = calc_outputs
        return self._calc_outputs
