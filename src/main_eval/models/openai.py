"""
utils for OpenAI models
"""

from __future__ import annotations 

import base64 
import mimetypes 
import re 
import os 
from pathlib import Path
from typing import Any

 
from openai import OpenAI 

from main_eval.dataset.prompt_builder import build_simple_selection_prompt
from main_eval.models.base import BaseVLM, ModelResponse 


class OpenAIModel: 
    def __init___(
        self,
        model_card: str = "gpt-5.2",
        max_output_tokens: int = 128,  
    ) -> None: 
        self.model_card = model_card 
        self.max_output_tokens = max_output_tokens 
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 
        
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
    
    def _image_to_data_url(self, image_path: str | Path) -> str: 
        image_path = Path(image_path) 
        if not image_path.exists(): 
            raise FileNotFoundError(f"Image not found: {image_path}") 
        
        mime_type, _ = mimetypes.guess_type(image_path) 
        if mime_type is None: 
            mime_type = "image/png" 
            
        with image_path.open("rb") as f: 
            encoded = base64.b64encode(f.read()).decode("utf-8") 
            
        return f"data: {mime_type};base64,{encoded}" 