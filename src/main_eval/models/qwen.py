from __future__ import annotations 

import re 
import os 
from pathlib import Path 
from openai import OpenAI 
from typing import Any 
from main_eval.dataset.prompt_builder import build_simple_selection_prompt
from main_eval.models.base import BaseVLM, ModelResponse


class QwenModel:
    def __init__(
        self,
        model_card: str = "Qwen/Qwen3.5-4B", 
        max_output_toknes: int = 64, 
    ) -> None: 
        self.model_card = model_card 
        self.max_output_tokens = max_output_toknes 
        #with open("../../../../data/cle.txt", "r") as f:
        #    key = f.read().strip()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) 
        print(os.environ["OPENAI_API_KEY"]) 
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
    
    def predict(self, sample: dict[str, Any]) -> ModelResponse:
        image_path = Path(sample["image_path"]) 
        if not image_path.exists(): 
            raise FileNotFoundError(f"Image not found: {image_path}") 
        
        prompt = build_simple_selection_prompt(sample) 
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_card,
            messages=messages,
            max_tokens=self.max_output_tokens,
            temperature=1.0, 
        ) 
        
        answer_text = response 
        predicted_option = self.parse_answer(answer_text) 
        
        return ModelResponse(
            predicted_option=predicted_option, 
            raw_response=response, 
        )