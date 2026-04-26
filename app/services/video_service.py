"""Video generation service: Replicate image-to-video and FFmpeg concatenation."""

import logging
import os
from pathlib import Path

import httpx
import replicate

from app.schemas.video import VideoScene

logger = logging.getLogger(__name__)


class VideoServiceError(Exception):
    pass


class VideoService:
    def __init__(self, replicate_token: str, video_model: str) -> None:
        self._replicate_token = replicate_token
        self._video_model = video_model

    async def generate_clip(
        self,
        image_path: str,
        scene: VideoScene,
        output_dir: str,
        clip_index: int,
    ) -> str:
        os.environ["REPLICATE_API_TOKEN"] = self._replicate_token
        rep_client = replicate.Client(api_token=self._replicate_token)  # type: ignore[attr-defined]

        with open(image_path, "rb") as img_file:
            output_url = await rep_client.async_run(
                self._video_model,
                input={
                    "prompt": scene.prompt,
                    "first_frame_image": img_file,
                    "duration": scene.duration_seconds,
                },
            )

        if isinstance(output_url, list):
            output_url = str(output_url[0])
        else:
            output_url = str(output_url)

        async with httpx.AsyncClient(timeout=120.0) as http_client:
            response = await http_client.get(output_url)
            response.raise_for_status()

            clip_path = str(Path(output_dir) / f"clip_{clip_index:02d}.mp4")
            with open(clip_path, "wb") as f:
                f.write(response.content)

        logger.info(f"Clip {clip_index} generated: {clip_path}")
        return clip_path

    def concatenate_clips(self, clip_paths: list[str], output_path: str) -> str:
        raise NotImplementedError
