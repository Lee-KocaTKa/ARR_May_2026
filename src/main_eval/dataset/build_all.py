from __future__ import annotations

from typing import Any

from main_eval.const import CATEGORY_DATASET_CONFIG, CATEGORY_ORDER 
from main_eval.dataset.loader import load_groups 
from main_eval.dataset.transform import build_vilstrub_samples 


def build_all_vilstrub_samples(text_field: str = "Meaning") -> list[dict[str, Any]]: 
    all_samples: list[dict[str, Any]] = [] 
    
    for category in CATEGORY_ORDER: 
        config = CATEGORY_DATASET_CONFIG[category] 
        groups = load_groups(config["json_path"]) 
        sample = build_vilstrub_samples(
            groups=groups, 
            category=category, 
            image_dir=config["image_dir"], 
            text_field=text_field, 
        )
        all_samples.extend(sample) 
        
    return all_samples 