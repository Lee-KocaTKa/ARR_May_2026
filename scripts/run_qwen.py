from __future__ import annotations 

import argparse 

from main_eval.const import CATEGORY_DATASET_CONFIG 
from main_eval.dataset.loader import load_groups
from main_eval.dataset.transform import build_vilstrub_samples
from main_eval.evaluation.runner import run_evalution_resumeable 
from main_eval.models.qwen import QwenModel



def main() -> None:
    parser = argparse.ArgumentParser() 
    parser.add_argument("--category", required=True, type=str)  
    args = parser.parse_args() 
    
    category = args.category 
    if category not in CATEGORY_DATASET_CONFIG: 
        raise ValueError(f"Invalid category: {category}. Must be one of {list(CATEGORY_DATASET_CONFIG.keys())}")
    
    config = CATEGORY_DATASET_CONFIG[category]
    
    groups = load_groups(config["json_path"]) 
    samples = build_vilstrub_samples(
        groups=groups,
        category=category,
        image_dir=config["image_dir"],
        text_field="Meaning",
    )
    
    model = QwenModel()
    
    output_path = f"outputs/by_cateogory/{category}_qwen8B_simple_selection.jsonl"
    
    print(f"Category: {category}")
    print(f"Groups: {len(groups)}")
    print(f"Samples: {len(samples)}")
    print(f"Output path: {output_path}")
    
    run_evalution_resumeable(
        model=model,
        samples=samples,
        output_path=output_path,
        log_every=10,
        fsync_every=1,
    )
    
    
if __name__ == "__main__": 
    main()