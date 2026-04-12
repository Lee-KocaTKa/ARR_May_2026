"""
A helper that writes and flushes immediately 
"""

from __future__ import annotations

import json 
import os 
from pathlib import Path
from typing import Any, TextIO 


def open_jsonl_append(path: str | Path) -> TextIO: 
    path = Path(path) 
    path.parent.mkdir(parents=True, exist_ok=True) 
    return path.open("a", encoding="utf-8")

def append_jsonl_record(f: TextIO, record: dict[str, Any], do_fsync: bool = False) -> None: 
    f.write(json.dumps(record, ensure_ascii=False) + "\n") 
    f.flush() 
    
    if do_fsync: 
        os.fsync(f.fileno())
        

 