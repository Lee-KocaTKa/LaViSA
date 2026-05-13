from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from main_eval.dataset.prompt_builder import build_simple_selection_prompt
from main_eval.models.base import ModelResponse

class GeminiModel:
    def __init__(
        self,
        model_card: str = "gemini-3.1-pro-preview",
        max_output_tokens: int = 128,
        api_key: str | None = None,
        use_json_output: bool = False,
        max_retries: int = 5,
    ) -> None:
        self.model_card = model_card
        self.max_output_tokens = max_output_tokens
        self.use_json_output = use_json_output
        self.max_retries = max_retries

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key is None:
            key = input("Enter your Gemini API key: ").strip()

        self.client = genai.Client(api_key=key)

    def parse_answer(self, text: str) -> str | None:
        text = text.strip()

        # If using JSON output, first try {"answer": "1"} or {"answer": 1}
        try:
            obj = json.loads(text)
            answer = obj.get("answer") or obj.get("predicted_option")
            if answer is not None:
                return str(answer)
        except json.JSONDecodeError:
            pass

        patterns = [
            r"Answer\s*:\s*(\d+)",
            r"^\s*(\d+)\s*$",
            r"\boption\s*(\d+)\b",
            r"\bOption\s*#?\s*(\d+)\b",
            r"\banswer\s+is\s+(\d+)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        return None

    def _image_to_part(self, image_path: str | Path) -> types.Part:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/png"

        return types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type=mime_type,
        )

    def _build_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
        }

        # Recommended for stable parsing, but you can keep False first
        # if you want to preserve the exact same prompt style as OpenAI/Qwen.
        if self.use_json_output:
            config["response_format"] = {
                "text": {
                    "mime_type": "application/json",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The selected option number, such as 1, 2, or 3.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A brief explanation for the choice.",
                            },
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            }

        return config

    def predict(self, sample: dict[str, Any]) -> ModelResponse:
        prompt = build_simple_selection_prompt(sample)
        image_part = self._image_to_part(sample["image_path"])

        last_error: str | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_card,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                image_part,
                            ],
                        )
                    ],
                    config=self._build_config(),
                )

                raw_text = (response.text or "").strip()
                predicted_option = self.parse_answer(raw_text)

                return {
                    "predicted_option": predicted_option,
                    "raw_text": raw_text,
                }

            except Exception as e:
                last_error = repr(e)
                time.sleep(min(60, 2**attempt))

        return {
            "predicted_option": None,
            "raw_text": "",
            "error": last_error,
        }