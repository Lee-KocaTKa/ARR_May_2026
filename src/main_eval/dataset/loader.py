from __future__ import annotations 

import json 
from pathlib import Path 
from typing import Iterator, Any 


def load_groups(path: str | Path) -> list[dict[str, Any]]: 
    path = Path(path) 
    with path.open("r", encoding="utf-8") as f: 
        data = json.load(f) 
    
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of groups in the JSON file at {path}, but got {type(data)}")
    
    return data 
