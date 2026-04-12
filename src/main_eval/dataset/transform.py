from __future__ import annotations

from pathlib import Path
from typing import Any 


def infer_category_from_group_id(group_id: str) -> str:
    """Infer the category of a sample based on its group ID."""
    return group_id.split("-")[0] 


def build_vilstrub_samples(
    groups: list[dict[str, Any]],
    category: str, 
    image_dir: str | Path, 
    text_field: str = "Meaning" 
) -> list[dict[str, Any]]: 
    image_dir = Path(image_dir) 
    samples: list[dict[str, Any]] = [] 
    
    for group in groups: 
        group_id = group["GroupID"] 
        ambiguous_caption = group["Sentence"] 
        variants = group["Variants"] 
        style = group.get("Style", None) 
        
        if len(variants) < 2: 
            continue 
        
        options = [variant[text_field] for variant in variants]  
        option_sentence_ids = [variant["SentenceID"] for variant in variants] 
        #descriptions = [variant["Description"] for variant in variants] 
        
        for gold_idx, gold_variant in enumerate(variants): 
            sample = {
                "sample_id": f"{group_id}__gold_{gold_idx+1}", 
                "group_id": group_id, 
                "category": category, 
                "style": style, 
                "ambiguous_caption": ambiguous_caption, 
                "image_path": str(image_dir / gold_variant["Image"]), 
                "options": options,
                "option_sentence_ids": option_sentence_ids,
                "image_description": gold_variant["Description"],
                "gold_option": str(gold_idx + 1),
                "gold_sentence_id": gold_variant["SentenceID"],  
            }
            samples.append(sample) 
            
    samples.sort(key=lambda x: x["sample_id"]) 
    return samples