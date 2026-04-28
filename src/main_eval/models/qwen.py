from __future__ import annotations 

import re 
import os 
from pathlib import Path 
from PIL import Image

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor 

from typing import Any 
from main_eval.dataset.prompt_builder import build_simple_selection_prompt
from main_eval.models.base import BaseVLM, ModelResponse


class QwenModel:
    def __init__(
        self,
        model_card: str = "Qwen/Qwen3-VL-32B-Instruct", 
        max_output_toknes: int = 64, 
    ) -> None: 
        self.model_card = model_card 
        self.max_output_tokens = max_output_toknes 
        #with open("../../../../data/cle.txt", "r") as f:
        #    key = f.read().strip()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_card, 
            dtype="auto", 
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_card) 
        #print(os.environ["OPENAI_API_KEY"]) 
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
        
        pil_image = Image.open(str(image_path)).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device) 
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids) 
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_toknes=True, 
            clean_up_tokenization_spaces=False
        )[0] 
        

        predicted_option = self.parse_answer(output_text) 
        
        return {
            "raw_text": output_text,
            "predicted_option": predicted_option,
        }