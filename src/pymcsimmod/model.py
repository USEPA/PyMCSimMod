from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

##### Variables / Constants


class Identifier(BaseModel):
    name: str


class DtVariable(BaseModel):
    identifier: Identifier


Variable = Identifier | DtVariable


class Number(BaseModel):
    value: float


##### Mathematical functions


class PowFunction(BaseModel):
    func: Literal["pow"] = "pow"
    args: list[Expression]


MathematicalFunction = PowFunction

##### Special functions
# TODO


##### Expressions


class ParenthesizedExpression(BaseModel):
    expression: Expression


class MathematicalExpression(BaseModel):
    operator: str
    lhs: Expression
    rhs: Expression


class NegativeExpression(BaseModel):
    expression: Variable | Number | ParenthesizedExpression


class Condition(BaseModel):
    operator: str
    lhs: Expression
    rhs: Expression


class TernaryExpression(BaseModel):
    condition: Condition
    if_true: Expression
    if_false: Expression


Expression = (
    NegativeExpression
    | MathematicalFunction
    | MathematicalExpression
    | TernaryExpression
    | Variable
    | Number
    | ParenthesizedExpression
)

##### Sections


class Statement(BaseModel):
    lhs: Variable
    rhs: Expression


class StatesSection(BaseModel):
    type: Literal["States"] = "States"
    declarations: list[Variable]


class InputsSection(BaseModel):
    type: Literal["Inputs"] = "Inputs"
    declarations: list[Variable]


class OutputsSection(BaseModel):
    type: Literal["Outputs"] = "Outputs"
    declarations: list[Variable]


class InitializeSection(BaseModel):
    type: Literal["Initialize"] = "Initialize"
    statements: list[Statement]


class DynamicsSection(BaseModel):
    type: Literal["Dynamics"] = "Dynamics"
    statements: list[Statement]


class JacobianSection(BaseModel):
    type: Literal["Jacobian"] = "Jacobian"
    statements: list[Statement]


class CalcOutputsSection(BaseModel):
    type: Literal["CalcOutputs"] = "CalcOutputs"
    statements: list[Statement]


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
