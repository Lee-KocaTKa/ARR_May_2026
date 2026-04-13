from __future__ import annotations

import json 
from pathlib import Path
from collections import defaultdict 

from main_eval.const import CATEGORY_ORDER


def load_jsonl(path: str | Path) -> list[dict]: 
    path = Path(path) 
    rows = [] 
    
    if not path.exists(): 
        return rows 
    
    with path.open("r", encoding="utf-8") as f: 
        for line_num, line in enumerate(f, start=1): 
            line = line.strip() 
            if not line: 
                continue 
            try:
                rows.append(json.loads(line)) 
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSONL line {line_num} in {path}") 
                
    return rows


def main() -> None: 
    result_dir = Path("outputs/by_cateogory") 
    
    total = 0 
    correct = 0
    parse_fail = 0 
    unsure = 0 
    
    per_category_total = defaultdict(int)
    per_category_correct = defaultdict(int)
    per_category_parse_fail = defaultdict(int)
    per_category_unsure = defaultdict(int)
    
    seen_ids = set() 
    
    for category in CATEGORY_ORDER: 
        path = result_dir / f"{category}_llava_onevision_simple_selection.jsonl"
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        for row in rows: 
            #print(row)
            sample_id = row["sample_id"]
            if sample_id in seen_ids:
                continue 
            seen_ids.add(sample_id) 
        
            total += 1 
            if row["is_correct"]:  
                #print("correct")
                correct += 1 
                per_category_correct[category] += 1
            else: 
                #print("incorrect")
                pass
            if row["predicted_option"] is None: 
                parse_fail += 1
                per_category_parse_fail[category] += 1
            
            per_category_total[category] += 1 
            #per_category_correct[category] += int(row["is_correct"])
            #per_category_parse_fail[category] += int(row["predicted_option"] is None)
        
    def acc(c: int, t: int) -> float: 
        return c / t if t > 0 else 0.0 
    
    #print("total unique samples across all categories:", total)
    
    
    print("\n=== Overall ===")
    print(
        f"total={total} correct={correct} "
        f"acc={acc(correct, total):.4f} "
        f"parse_fail={acc(parse_fail, total):.4f}"
    )

    print("\n=== Per category ===")
    for category in CATEGORY_ORDER:
        t = per_category_total[category]
        c = per_category_correct[category]
        p = per_category_parse_fail[category]
        print(
            f"{category:<8} total={t:<4} correct={c:<4} "
            f"acc={acc(c, t):.4f} parse_fail={acc(p, t):.4f}"
        )


if __name__ == "__main__":
    main()