"""
Celery tasks for image processing pipeline.

This module contains image processing tasks that route based on mode:
- remove_bg: Background Removal only (RMBG-1.4)
- edit: Instruction-based image editing (FireRed-Image-Edit-1.1)
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

from celery import Task

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.schemas.task import TaskMode, TaskStatus
from app.services.ai_service import AIServiceFactory
from app.services.storage import GCSStorage, LocalStorage, StorageService

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Helper to run async code in sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class ImageProcessingTask(Task):  # type: ignore[misc]
    """Base task class for image processing with status updates."""

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status in the task store."""
        self.update_state(state=status.value, meta={"task_id": task_id})


@celery_app.task(
    bind=True,
    base=ImageProcessingTask,
    name="app.tasks.image_processing.process_image",
)
def process_image(
    self: ImageProcessingTask,
    task_id: str,
    image_path: str,
    mode: str = "edit",
    instruction: Optional[str] = None,
) -> dict[str, str]:
    """
    Main image processing task (Hybrid GCS/Local Support).

    Routes based on mode:
    - "remove_bg": Background removal only (RMBG-1.4)
    - "edit": Instruction-based editing (FireRed-Image-Edit-1.1)
    """
    settings = get_settings()
    storage: StorageService
    if image_path.startswith("gs://") or settings.storage_type == "gcs":
        storage = GCSStorage(bucket_name=settings.gcs_bucket_name)
    else:
        storage = LocalStorage(base_dir=Path(settings.local_storage_path))

    # Create a local temporary directory for this task
    temp_dir = Path(tempfile.mkdtemp(prefix=f"task_{task_id}_"))
    local_input_path = temp_dir / Path(image_path).name

    try:
        # 1. Download source image if it's remote or just copy it to temp
        if image_path.startswith("gs://") or image_path.startswith("http"):
            gcs_key = image_path.replace(f"gs://{settings.gcs_bucket_name}/", "")
            run_async(storage.download(gcs_key, str(local_input_path)))
        else:
            shutil.copy2(image_path, local_input_path)

        task_mode = TaskMode(mode)

        if task_mode == TaskMode.REMOVE_BG:
            # --- Background Removal Only ---
            self.update_task_status(task_id, TaskStatus.REMOVING_BG)
            bg_service = AIServiceFactory.get_background_removal_service()
            local_result = run_async(bg_service.process(str(local_input_path)))

            # Upload result
            result_dest_key = f"processed/{task_id}/bg_removed.png"
            result_url = run_async(storage.upload(local_result, result_dest_key))

        elif task_mode == TaskMode.EDIT:
            # --- Instruction-based Editing (FireRed-Image-Edit-1.1) ---
            self.update_task_status(task_id, TaskStatus.EDITING)
            edit_service = AIServiceFactory.get_firered_edit_service()
            edit_instruction = instruction or (
                "professional product photography, clean white studio background, "
                "soft studio lighting, sharp focus, photorealistic"
            )
            local_result = run_async(
                edit_service.process(
                    str(local_input_path),
                    instruction=edit_instruction,
                )
            )

            # Upload result
            result_dest_key = f"processed/{task_id}/edited.png"
            result_url = run_async(storage.upload(local_result, result_dest_key))

        else:
            raise ValueError(f"Unknown task mode: {mode}")

        # Mark as completed
        self.update_task_status(task_id, TaskStatus.COMPLETED)

        return {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED.value,
            "result_url": result_url,
        }

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        self.update_task_status(task_id, TaskStatus.FAILED)
        return {
            "task_id": task_id,
            "status": TaskStatus.FAILED.value,
            "error": error_msg,
        }
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
