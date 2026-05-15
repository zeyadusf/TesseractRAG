from enum import Enum

class EvaluationTech(str, Enum):
    LLMS = "LLMs Guide"
    DEFAULT = LLMS