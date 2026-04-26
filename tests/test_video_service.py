"""Unit tests for VideoService."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.video import VideoScene
from app.services.video_service import VideoService


@pytest.fixture
def sample_scene() -> VideoScene:
    return VideoScene(
        id=1,
        shot_type="Close-up",
        camera_motion="Static shot",
        visual_description="Product on white surface",
        duration_seconds=5,
        prompt="Close-up, static shot. Product centered on white surface...",
    )


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "product.png"
    img.save(str(path))
    return str(path)


class TestGenerateClip:
    @pytest.mark.asyncio
    async def test_generate_clip_returns_path(
        self,
        sample_image_path: str,
        sample_scene: VideoScene,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        mock_video_bytes = b"fake_mp4_bytes"

        with (
            patch("app.services.video_service.replicate") as mock_replicate,
            patch("app.services.video_service.httpx") as mock_httpx,
        ):
            mock_replicate.Client.return_value.async_run = AsyncMock(
                return_value="https://replicate.delivery/fake/clip.mp4"
            )
            mock_response = MagicMock()
            mock_response.content = mock_video_bytes
            mock_response.raise_for_status = MagicMock()
            mock_httpx.AsyncClient.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            service = VideoService(
                replicate_token="test-token",
                video_model="minimax/video-01",
            )
            result_path = await service.generate_clip(
                image_path=sample_image_path,
                scene=sample_scene,
                output_dir=str(output_dir),
                clip_index=0,
            )

        assert result_path.endswith(".mp4")
        assert os.path.exists(result_path)
        assert open(result_path, "rb").read() == mock_video_bytes

    @pytest.mark.asyncio
    async def test_generate_clip_raises_on_replicate_error(
        self,
        sample_image_path: str,
        sample_scene: VideoScene,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "clips"
        output_dir.mkdir()

        with patch("app.services.video_service.replicate") as mock_replicate:
            mock_replicate.Client.return_value.async_run = AsyncMock(
                side_effect=Exception("Replicate error")
            )
            service = VideoService(
                replicate_token="test-token",
                video_model="minimax/video-01",
            )
            with pytest.raises(Exception, match="Replicate error"):
                await service.generate_clip(
                    image_path=sample_image_path,
                    scene=sample_scene,
                    output_dir=str(output_dir),
                    clip_index=0,
                )
