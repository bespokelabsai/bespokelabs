"""Defines the QAC data example"""

import json
import random
from pydantic import BaseModel
from typing import TypeVar, Generic, List


def remove_markdown(output):
    output = output.strip()
    if output.startswith("```json"):
        output = output[7:]
    if output.endswith("```"):
        output = output[:-3]
    return output


class QAC(BaseModel):
    """Class for a single QAC example."""

    question: str
    answer: str
    rating: float = None
    context: list[str] = []


# Define a TypeVar for types that are subclasses of QAC
QACType = TypeVar("QACType", bound=QAC)


class QACs(Generic[QACType]):
    """Class for  QACs."""

    qacs: List[QACType]

    def __init__(self, qacs: List[QACType]):
        self.qacs = qacs

    def sample(self, n: int) -> "QACs[QACType]":
        return QACs(random.sample(self.qacs, n))

    def to_json(self) -> str:
        out = [qac.model_dump() for qac in self.qacs]
        return json.dumps(out, indent=2)

    def __add__(self, other: "QACs[QACType]") -> "QACs[QACType]":
        return QACs(self.qacs + other.qacs)
