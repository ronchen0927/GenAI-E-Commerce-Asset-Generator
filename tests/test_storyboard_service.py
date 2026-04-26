"""Unit tests for StoryboardService."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.schemas.video import StoryboardResponse
from app.services.storyboard_service import StoryboardService


@pytest.fixture
def sample_image_path(tmp_path: pytest.TempPathFactory) -> str:
    img = Image.new("RGB", (100, 100), color="blue")
    path = tmp_path / "product.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def mock_openai_response() -> MagicMock:
    import json

    payload = {
        "product_summary": "Blue product",
        "scenes": [
            {
                "id": 1,
                "shot_type": "Extreme macro close-up",
                "camera_motion": "Slow dolly forward",
                "visual_description": "Blue surface texture",
                "duration_seconds": 5,
                "prompt": "Extreme macro close-up, slow dolly forward. Blue surface...",
            }
        ],
    }
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(payload)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestStoryboardService:
    @pytest.mark.asyncio
    async def test_generate_returns_storyboard(
        self,
        sample_image_path: str,
        mock_openai_response: MagicMock,
    ) -> None:
        with patch("app.services.storyboard_service.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai_cls.return_value = mock_client

            service = StoryboardService(api_key="test-key")
            result = await service.generate(
                image_path=sample_image_path,
                image_storage_path="storage/abc/original.png",
                style="cinematic",
                num_scenes=1,
            )

        assert isinstance(result, StoryboardResponse)
        assert result.source == "ai"
        assert len(result.scenes) == 1
        assert result.image_path == "storage/abc/original.png"

    @pytest.mark.asyncio
    async def test_generate_falls_back_to_template_on_openai_error(
        self,
        sample_image_path: str,
    ) -> None:
        with patch("app.services.storyboard_service.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("API down")
            mock_openai_cls.return_value = mock_client

            service = StoryboardService(api_key="test-key")
            result = await service.generate(
                image_path=sample_image_path,
                image_storage_path="storage/abc/original.png",
                style="cinematic",
                num_scenes=3,
            )

        assert result.source == "template"
        assert len(result.scenes) == 3
        assert result.image_path == "storage/abc/original.png"

    @pytest.mark.asyncio
    async def test_generate_falls_back_on_invalid_json(
        self,
        sample_image_path: str,
    ) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "not valid json {{{"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("app.services.storyboard_service.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            service = StoryboardService(api_key="test-key")
            result = await service.generate(
                image_path=sample_image_path,
                image_storage_path="storage/abc/original.png",
                style="cinematic",
                num_scenes=3,
            )

        assert result.source == "template"
