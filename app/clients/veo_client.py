from __future__ import annotations

import time
from io import BytesIO
from typing import Tuple

import anyio


class VeoClient:
    def __init__(self, _httpx_client):
        self._httpx_client = _httpx_client

    def _generate_sync(self, api_key: str, prompt: str, image_bytes: bytes, image_mime: str, model: str) -> Tuple[bytes, str]:
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise RuntimeError("google-genai is required. Add 'google-genai' to dependencies.") from e

        client = genai.Client(api_key=api_key)

        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        ext = ext_map.get(image_mime, ".png")

        class NamedBytesIO(BytesIO):
            def __init__(self, data: bytes, name: str):
                super().__init__(data)
                self.name = name

        file_obj = NamedBytesIO(image_bytes, f"upload{ext}")

        try:
            upload = client.files.upload(file=file_obj, mime_type=image_mime)
        except TypeError:
            try:
                upload = client.files.upload(file=file_obj)
            except Exception as e:
                raise

        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=upload,
        )

        while not operation.done:
            time.sleep(3)
            operation = client.operations.get(operation)

        video = operation.response.generated_videos[0]
        blob = client.files.download(file=video.video)

        if hasattr(blob, "read"):
            data = blob.read()
        elif isinstance(blob, (bytes, bytearray)):
            data = bytes(blob)
        elif hasattr(blob, "data"):
            data = blob.data  # type: ignore[attr-defined]
        else:
            raise RuntimeError("Unexpected download type from google-genai")

        return data, "video/mp4"

    async def generate_video(self, api_key: str, prompt: str, image_bytes: bytes, image_mime: str, model: str) -> Tuple[bytes, str]:
        return await anyio.to_thread.run_sync(self._generate_sync, api_key, prompt, image_bytes, image_mime, model)
