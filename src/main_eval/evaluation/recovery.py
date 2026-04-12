"""
This is for when the job is terminated in the middle
"""

from __future__ import annotations 

import json 
from pathlib import Path


def load_completed_sample_ids(path: str | Path) -> set[str]: 
    path = Path(path) 
    
    if not path.exists():
        return set()
    
    completed: set[str] = set() 
    
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1): 
            line = line.strip() 
            if not line: 
                continue 
            
            try: 
                row = json.loads(line) 
            except json.JSONDecodeError: 
                # Ignore a possibly truncated last line from interrupted jobs 
                print(f"Warning: skipping malformed JSONL line {line_num} in {path}") 
                continue 
            
            sample_id = row.get("sample_id") 
            if sample_id is not None: 
                completed.add(sample_id) 
                
    return completed