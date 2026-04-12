from __future__ import annotations 

from abc import ABC, abstractmethod
from dataclasses import dataclass 
from typing import Any 



@dataclass 
class ModelResponse: 
    """A dataclass to represent the response from a model."""
    raw_text: str
    predicted_option: int | None 
    

class BaseVLM(ABC): 
    @abstractmethod
    def predict(self, sample: dict[str, Any]) -> ModelResponse: 
        """Predict the label for a given sample."""
        raise NotImplementedError()  