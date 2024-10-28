from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum
import uuid
from pydantic import BaseModel, Field, model_validator


class RunStepStatus(Enum):
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStepActionType(Enum):
    ACTION = "action"
    THOUGHT = "thought"
    OBSERVE = "observe"
    COMPLETE = "complete"


class RunStepCallbackMessage(BaseModel):
    """
    Base model for run callback message
    """
    id: str = Field(default_factory=lambda: uuid.uuid4().hex,
                    description="id for run step")
    step_type: RunStepActionType = Field(
        ...,
        description="type of the step taken"
    )
    status: RunStepStatus = Field(
        RunStepStatus.NOT_STARTED,
        description="status of run step")
    started_at: str = Field(default_factory=lambda: datetime.now(
    ).isoformat(), description="start datetime in ISO format")
    completed_at: Optional[str] = Field(
        None,
        description="end datetime in ISO format")
    args: Optional[Dict[str, Any]] = Field(
        None,
        description="arguments for action"
    )
    content: str = Field(
        "",
        description="text content of llm action when thought, observe, completion"

    )
    tool_used: str = Field(
        None,
        description="name of the tool used"
    )
