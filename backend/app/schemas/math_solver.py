"""Internal, validated contracts for symbolic tools; never executable code."""
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

MathOperation = Literal["calculate", "solve_equation", "differentiate", "integrate", "simplify"]


class MathTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: Literal["normal", "math_problem"] = "math_problem"
    operation: MathOperation | None = None
    expression: str | None = Field(default=None, max_length=512)
    left: str | None = Field(default=None, max_length=512)
    right: str | None = Field(default=None, max_length=512)
    variable: str | None = Field(default=None, pattern=r"^[a-zA-Z]$")
    source: Literal["user_question", "retrieved_document"] = "user_question"
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_chunk_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_fields(self):
        if self.intent == "normal":
            if any((self.operation, self.expression, self.left, self.right, self.variable)):
                raise ValueError("Normal tasks cannot contain solver fields")
        elif self.operation == "solve_equation":
            if not self.left or not self.right or self.expression:
                raise ValueError("Equations require only left and right expressions")
        elif not self.operation or not self.expression or self.left or self.right:
            raise ValueError("Operation requires a single expression")
        return self


class MathSolverResult(BaseModel):
    success: bool
    problem_type: MathOperation | None = None
    expression: str | None = None
    result: str | list[str] | None = None
    variable: str | None = None
    error: str | None = None
    error_type: str | None = None
    notes: list[str] = Field(default_factory=list)


class SolverMetadata(BaseModel):
    used: bool = False
    type: Literal["math"] = "math"
    operation: MathOperation | None = None
    verified: bool = False
