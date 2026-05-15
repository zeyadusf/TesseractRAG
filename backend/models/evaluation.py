from pydantic import BaseModel, Field,model_validator
from typing import Optional, List
from uuid import UUID


#  SHARED / INTERNAL MODELS
class Metrics(BaseModel):

    faithfulness: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevancy: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    context_precision: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    context_recall: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)


class EvaluationItem(BaseModel):
    message_id: UUID
    metrics: Metrics


#  REQUEST MODELS 
class EvaluateMessageRequest(BaseModel):
    """Request to evaluate a SINGLE assistant message."""
    message_id: UUID = Field(..., description="ID of the assistant message to evaluate")
    session_id: UUID = Field(..., description="Session ID for ownership validation")
    model: Optional[str] = Field(default=None, description="Optional: override evaluator model (e.g. 'command-r-plus')")


class EvaluateSessionRequest(BaseModel):
    """Request to batch-evaluate ALL assistant messages in a session."""
    session_id: UUID = Field(..., description="Session ID to evaluate")
    model: Optional[str] = Field(default=None, description="Optional: override evaluator model")
    limit: int = Field(default=200, ge=1, le=1000, description="Max messages to evaluate")


# RESPONSE MODELS 

class EvaluationOut(BaseModel):

    id: UUID
    message_id: UUID
    session_id: UUID
    
    # Metrics (nullable until evaluated)
    faithfulness: Optional[float] = Field(None, ge=0.0, le=100.0)
    answer_relevancy: Optional[float] = Field(None, ge=0.0, le=100.0)
    context_precision: Optional[float] = Field(None, ge=0.0, le=100.0)
    context_recall: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Metadata
    evaluator: str = Field(..., description="Evaluator provider (e.g. 'cohere')")
    eval_model: Optional[str] = Field(None, description="Model used for evaluation")
    
    model_config = {"from_attributes": True}

    # @model_validator(mode="before")
    # @classmethod
    # def convert_to_percentage(cls, values):
    #     for field in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
    #         v = values.get(field)
    #         if v is not None and v <= 1.0:  # بيتحول بس لو بين 0-1
    #             values[field] = round(v * 100, 1)
    #     return values


class SessionEvaluationSummary(BaseModel):

    session_id: UUID
    avg_faithfulness:  Optional[float] = Field(None, ge=0.0, le=100.0)
    avg_answer_relevancy:  Optional[float] = Field(None, ge=0.0, le=100.0)
    avg_context_precision:  Optional[float] = Field(None, ge=0.0, le=100.0)
    avg_context_recall:  Optional[float] = Field(None, ge=0.0, le=100.0)
    total_evaluated: int = Field(..., description="Number of messages that have been evaluated")


class EvaluationResponse(BaseModel):

    count: int = Field(..., description="Total items returned")
    avg_metrics: Optional[Metrics] = Field(default=None, description="Averaged metrics across items")
    history: List[EvaluationItem] = Field(default_factory=list, description="Detailed evaluation history")