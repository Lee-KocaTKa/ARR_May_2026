from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

from main_eval.dataset.prompt_builder import build_simple_selection_prompt 
from main_eval.models.base import BaseVLM, ModelResponse 

class LlavaOneVisionModel:
    def __init__(
        self,
        model_path: str = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )


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
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        predicted_option = self.parse_answer(output_text)

        return {
            "raw_text": output_text,
            "predicted_option": predicted_option,
        }