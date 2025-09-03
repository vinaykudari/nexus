from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from ..models import ChatRequest


class ChatProvider(ABC):
    name: str

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @abstractmethod
    async def chat(self, api_key: str, req: ChatRequest) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def stream(self, api_key: str, req: ChatRequest) -> AsyncIterator[bytes]:
        ...

