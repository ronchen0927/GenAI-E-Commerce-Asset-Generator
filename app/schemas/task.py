from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskMode(str, Enum):
    """Processing mode for the task."""

    REMOVE_BG = "remove_bg"  # Background removal only (RMBG-1.4)
    EDIT = "edit"  # Instruction-based image editing (FireRed-Image-Edit-1.1)


class TaskStatus(str, Enum):
    """Task processing status enum."""

    PENDING = "PENDING"
    REMOVING_BG = "REMOVING_BG"
    EDITING = "EDITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskCreate(BaseModel):
    """Schema for creating a new task."""

    mode: TaskMode = Field(
        TaskMode.EDIT, description="Processing mode: 'remove_bg' or 'edit'"
    )
    instruction: Optional[str] = Field(
        None,
        description=(
            "Editing instruction for FireRed"
            " (e.g. 'Place this product on a marble table')"
        ),
    )


class TaskResponse(BaseModel):
    """Schema for task response."""

    task_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    original_url: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None

    class Config:
        from_attributes = True


class TaskStatusResponse(BaseModel):
    """Schema for task status query response."""

    task_id: str
    status: TaskStatus
    progress: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Schema for upload response."""

    task_id: str
    message: str
    status: TaskStatus
