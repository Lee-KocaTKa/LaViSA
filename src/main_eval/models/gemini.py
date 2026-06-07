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
        max_output_tokens: int = 128
    ) -> None:
        self.model_card = model_card
        self.max_output_tokens = max_output_tokens
        key = input("Enter your Gemini API key: ").strip()

        self.client = genai.Client(api_key=key)

    def parse_answer(self, text: str) -> int | None:
        text = text.strip()

        patterns = [
            r"Answer\s*:\s*(\d+)",
            r"^\s*(\d+)\s*$",
            r"\boption\s*(\d+)\b",
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

        with open(image_path, "rb") as f: 
            image_bytes = f.read()

        return types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

    

    def predict(self, sample: dict[str, Any]) -> ModelResponse:
        prompt = build_simple_selection_prompt(sample)
        image_part = self._image_to_part(sample["image_path"])

        

        response = self.client.models.generate_content(
            model=self.model_card,
            contents=[
                image_part,
                prompt 
            ]
        )
        
        raw_text = response.text
        predicted_option = self.parse_answer(raw_text) 
        
        return {
            "predicted_option": predicted_option,
            "raw_text": raw_text,
        }
        