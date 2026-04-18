from __future__ import annotations

from typing import Any 


def build_simple_selection_prompt(sample: dict[str, Any]) -> str: 
    """Build a simple prompt for selection tasks."""
    # input : ambig sentence, image, question, options 
    # output : choose the write disambiguated option that matches the iamge 
    
    caption = sample["ambiguous_caption"] 
    options = sample["options"] 
    
    lines = [
        "You are a vision-language model performing structural disambiguation.",
        "",
        "You will be given an ambiguous caption and an image.",
        "Your task is to select the correct interpretation of the caption based on the image.",
        "",
        "Important:",
        "- All options are derived from the same ambiguous caption and differ only in structural interpretation.",
        "- You must use the image to decide.",
        "- Do not rely only on textual plausibility.",
        "",
        f'Caption: "{caption}"',
        "",
        "Question:",
        "Among the following options, which one correctly reflects the meaning of the caption given the image?",
    ]
    
    for i, option in enumerate(options, start=1):
        lines.append(f'{i}) "{option}"')
        
    #lines.append(f"{len(options) + 1}) I cannot decide")
    lines.append("")
    lines.append("Please provide your answer in the following format:")
    lines.append("Answer: <one of the options> as a single number (1, 2, 3, etc.)") 
    
    return "\n".join(lines) 