from __future__ import annotations

from collections import defaultdict 
from pathlib import Path 
from typing import Iterable 

from main_eval.models.base import BaseVLM
from main_eval.evaluation.recovery import load_completed_sample_ids 
from main_eval.evaluation.writer import open_jsonl_append, append_jsonl_record 


CATEGORY_ORDER = ["vp", "pp", "anaph", "ellip", "adj", "vb", "conj"]



def _safe_acc(correct: int, total: int) -> float: 
    return (correct / total * 100) if total > 0 else 0.0 


def run_evalution_resumeable(
    model: BaseVLM, 
    samples: Iterable[dict], 
    output_path: str | Path,  
    log_every: int = 20,
    fsync_every: int = 1,  
) -> dict: 
    output_path = Path(output_path)
    completed_ids = load_completed_sample_ids(output_path) 
    
    print(f"Found {len(completed_ids)} completed samples in {output_path}")  
    
    #if summary_path is not None: 
    #    summary_path = Path(summary_path)
    #    summary_path.parent.mkdir(parents=True, exist_ok=True) 
    
    total = 0 
    correct = 0 
    parse_fail = 0 
    skipped = 0 
    
    per_category_total: dict[str, int] = defaultdict(int)
    per_category_correct: dict[str, int] = defaultdict(int)
    per_category_parse_fail: dict[str, int] = defaultdict(int) 
    
    with open_jsonl_append(output_path) as f: 
        for sample in samples:
            # print the current progress (n out of total)
            print(f"Processing sample {total+1} / {len(samples)} (skipped {skipped})", end="\r") 
            
            
            sample_id = sample["sample_id"]
            
            if sample_id in completed_ids:
                skipped += 1 
                continue
             
            response = model.predict(sample) 
            
            pred = response["predicted_option"] 
            gold = sample["gold_option"] 
            category = sample["category"]
            
            
            
            is_correct = False if pred is None else (gold in pred) 
            
            total += 1
            correct += int(is_correct) 
            parse_fail += int(pred is None) 
            
            per_category_total[category] += 1
            per_category_correct[category] += int(is_correct)
            per_category_parse_fail[category] += int(pred is None)

            record = {
                "sample_id": sample["sample_id"], 
                "group_id": sample["group_id"], 
                "category": sample["category"],
                "style": sample["style"],
                "gold_option": gold,
                "predicted_option": pred,
                "is_correct": is_correct,
                "raw_text": response["raw_text"],
                "options": sample["options"],
                "image_path": sample["image_path"],
                "ambiguous_caption": sample["ambiguous_caption"], 
            }
            #f.write(json.dumps(record, ensure_ascii=False) + "\n") 
            
            do_fsync = (total % fsync_every == 0) 
            append_jsonl_record(f, record, do_fsync=do_fsync) 
            
            if total % log_every == 0:
                print(
                    f"Processed {total} samples (skipped {skipped}), "
                    f"[{total}] overall_acc={_safe_acc(correct, total):.4f} "
                    f"parse_fail={_safe_acc(parse_fail, total):.4f}"
                )
                
    # Build Summary
    category_names = [c for c in CATEGORY_ORDER if c in per_category_total] 
    remaining = [c for c in per_category_total if c not in CATEGORY_ORDER] 
    category_names.extend(sorted(remaining))  
    
    
    summary = {
        "newly_processed": total,
        "skipped": skipped, 
        "overall": {
            "total": total, 
            "correct": correct,
            "accuracy": _safe_acc(correct, total),
            "parse_fail": parse_fail,
            "parse_fail_rate": _safe_acc(parse_fail, total),
        },
        "per_category": {},
    }

    for category in category_names: 
        cat_total = per_category_total[category] 
        cat_correct = per_category_correct[category] 
        cat_parse_fail = per_category_parse_fail[category] 
        
        summary["per_category"][category] = {
            "total": cat_total, 
            "correct": cat_correct,
            "accuracy": _safe_acc(cat_correct, cat_total),
            "parse_fail": cat_parse_fail,
            "parse_fail_rate": _safe_acc(cat_parse_fail, cat_total),
        }

    # Print Summary 
    print("\n=== This Run Summary ===")
    print(
        f"newly_processed={summary['newly_processed']} "
        #f"skipped_existing={summary['skipped_existing']}"
    )
    print(
        f"overall: total={summary['overall']['total']} "
        f"correct={summary['overall']['correct']} "
        f"acc={summary['overall']['accuracy']:.4f}"
    )
    for category in category_names:
        item = summary["per_category"][category]
        print(
            f"  {category:<8} total={item['total']:<4} "
            f"correct={item['correct']:<4} acc={item['accuracy']:.4f}"
        )

    return summary