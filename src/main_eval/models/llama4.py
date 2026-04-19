from __future__ import annotations 

import re 
from typing import Any

from pathlib import Path 

import torch 
from PIL import Image 
from transformers import AutoProcessor, Llama4ForConditionalGeneration

from main_eval.dataset.prompt_builder import build_simple_selection_prompt 
from main_eval.models.base import BaseVLM, ModelResponse 


class Llama4VLM (BaseVLM): 
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        attn_implmentation: str = "flex_attention", 
        torch_dtype: torch.dtype = torch.float16, 
        device_map: str = "auto", 
        max_new_tokens: int = 512, 
    ) -> None: 
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        
        self.processor = AutoProcessor.from_pretrained(model_id) 
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_id, 
            attn_implementation=attn_implmentation, 
            torch_dtype=torch_dtype, 
            device_map=device_map, 
        ) 
        
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
                return int(match.group(1)) 
            
            
        return None
    
    def _load_image(self, image_path: str | Path) -> Image.Image: 
        image_path = Path(image_path)
        if not image_path.exists(): 
            raise FileNotFoundError(f"Image not found: {image_path}") 
        return Image.open(image_path).convert("RGB") 
    
    def predict(self, sample: dict[str, Any]) -> ModelResponse:
        """Predict the label for a given sample."""
        prompt = build_simple_selection_prompt(sample) 
        image = self._load_image(sample["image_path"]) 
        #image = sample["image_path"] 
        
        messages = [
            {
                "role": "user", 
                "content": [ 
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},  
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True, 
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device) 
        
        with torch.inference_mode(): 
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=False, 
            )
            
        generated_ids = outputs[:, inputs["inputs_ids"].shape[-1]:] 
        raw_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
        )[0].strip() 
        
        predicted_option = self._parse_answer(raw_text) 
        return ModelResponse(
            raw_text=raw_text, 
            predicted_option=predicted_option,
        ) 