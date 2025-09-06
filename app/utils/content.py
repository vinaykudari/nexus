from __future__ import annotations

import base64
import mimetypes
from typing import List, Union, Dict, Any

from ..models import ContentPart, TextPart, ImageURLPart, ImageBase64Part


def normalize_message_content(content: Union[str, List[ContentPart]]) -> List[ContentPart]:
    if isinstance(content, str):
        return [TextPart(type="text", text=content)]
    return content


def guess_mime_from_url(url: str) -> str:
    mime, _ = mimetypes.guess_type(url)
    return mime or "image/png"


def to_openai_message_content(parts: List[ContentPart]) -> Union[str, List[Dict[str, Any]]]:
    # If it is a single text part, keep legacy simple string for smaller payloads
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    out: List[Dict[str, Any]] = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, ImageURLPart):
            out.append({"type": "image_url", "image_url": {"url": p.url}})
        elif isinstance(p, ImageBase64Part):
            data_url = f"data:{p.mime_type};base64,{p.data}"
            out.append({"type": "image_url", "image_url": {"url": data_url}})
    return out


def to_anthropic_content_blocks(parts: List[ContentPart]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for p in parts:
        if isinstance(p, TextPart):
            blocks.append({"type": "text", "text": p.text})
        elif isinstance(p, ImageURLPart):
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "url", "url": p.url, "media_type": p.mime_type or guess_mime_from_url(p.url)},
                }
            )
        elif isinstance(p, ImageBase64Part):
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": p.mime_type, "data": p.data},
                }
            )
    return blocks


def parts_from_openai_message(message: Dict[str, Any]) -> List[ContentPart]:
    parts: List[ContentPart] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(TextPart(type="text", text=content))
        return parts
    if isinstance(content, list):
        for c in content:
            t = c.get("type") if isinstance(c, dict) else None
            if t == "text":
                txt = c.get("text")
                if isinstance(txt, str):
                    parts.append(TextPart(type="text", text=txt))
            elif t == "image_url":
                iu = c.get("image_url") or {}
                url = iu.get("url")
                if isinstance(url, str):
                    if url.startswith("data:") and ";base64," in url:
                        mime = url.split(":", 1)[1].split(";", 1)[0]
                        b64 = url.split(",", 1)[1]
                        parts.append(ImageBase64Part(type="image_base64", data=b64, mime_type=mime))
                    else:
                        parts.append(ImageURLPart(type="image_url", url=url))
    return parts


def parts_from_anthropic_content(blocks: List[Dict[str, Any]]) -> List[ContentPart]:
    out: List[ContentPart] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            out.append(TextPart(type="text", text=b["text"]))
        elif b.get("type") == "image" and isinstance(b.get("source"), dict):
            src = b["source"]
            st = src.get("type")
            if st == "base64":
                mime = src.get("media_type") or "image/png"
                data = src.get("data")
                if isinstance(data, str):
                    out.append(ImageBase64Part(type="image_base64", data=data, mime_type=mime))
            elif st == "url":
                url = src.get("url")
                mime = src.get("media_type") or guess_mime_from_url(url or "")
                if isinstance(url, str):
                    out.append(ImageURLPart(type="image_url", url=url, mime_type=mime))
    return out

