from __future__ import annotations 

import re 
from pathlib import Path
from typing import Any

import torch 
from transformers import AutoProcessor, Gemma3ForConditionalGeneration 
from PIL import Image 

from main_eval.dataset.prompt_builder import build_simple_selection_prompt 
from main_eval.models.base import BaseVLM, ModelResponse 
 
class GemmaModel: 
    def __init__(
        self, 
        model_path: str = "google/gemma-3-27b-it"
    ) -> None:
    
        self.model_path = model_path 
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_path) 
        
    def _parse_answer(self, text: str) -> int | None: 
        """Parse the model's answer to extract the predicted option number."""
        text = text.strip() 
        
        patterns = [
            r"Answer\s*:\s*(\d+)",
            r"^\s*(\d+)\s*$", 
            r"[Oo]ption\s*(\d+)",         
        ]
        
        for pattern in patterns: 
            match = re.search(pattern, text, flags=re.MULTILINE)
            if match: 
                return match.group(1) 
        
        return None  
    
    def predict(self, sample: dict[str, Any]) -> dict[str, Any]: 
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
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16) 
        
        input_len = inputs["input_ids"].shape[-1] 
        
        with torch.inference_mode(): 
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100, 
                do_sample=False, 
            )
            
            generation = outputs[0][input_len:] 
            
        decoded = self.processor.decode(
            generation,
            skip_special_tokens=True 
        )
        
        predicted_option = self._parse_answer(decoded) 
        
        return {
            "predicted_option": predicted_option,
            "raw_text": decoded,
        }
        
        