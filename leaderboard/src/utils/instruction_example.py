"""Defines the instruction tuning data example"""
import json
import random
from pydantic import BaseModel
from typing import TypeVar, Generic, List


class Example(BaseModel):
  """Class for a single instruction tuning example."""
  scores: dict = {}
  feedbacks: dict = {}


class QAC(Example):
  question: str
  answer: str
  rating: float = None
  context: list[str] = []
  self_instruct_round: int = 0
  improved_example: Example = None


# Define a TypeVar for types that are subclasses of Example
ExampleType = TypeVar('ExampleType', bound=Example)

class Examples(Generic[ExampleType]):
    """Class for multiple instruction tuning examples."""
    examples: List[ExampleType]

    def __init__(self, examples: List[ExampleType]):
        self.examples = examples

    def sample(self, n: int) -> 'Examples[ExampleType]':
        return Examples(random.sample(self.examples, n))

    def to_json(self) -> str:
        out = [example.model_dump() for example in self.examples]
        return json.dumps(out, indent=2)
    
    def __add__(self, other: 'Examples[ExampleType]') -> 'Examples[ExampleType]':
        return Examples(self.examples + other.examples)
