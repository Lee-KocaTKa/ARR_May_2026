from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

MODEL_PATH = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
IMAGE_PATH = Path("../../../data/ViLStrUB/images/vp/vp-1-a-i.png") 


def main() -> None:
    print("=== CUDA INFO ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} / {props.total_memory / (1024**3):.2f} GB")
    print()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print("=== LOADING MODEL ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded.")

    print("=== LOADING PROCESSOR ===")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    print("Processor loaded.\n")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(IMAGE_PATH),
                },
                {
                    "type": "text",
                    "text": "Describe this image in one short sentence.",
                },
            ],
        }
    ]

    print("=== PREPARING INPUTS ===")
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    print()

    print("=== GENERATING ===")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("=== OUTPUT ===")
    print(output_text[0])
    print()

    if torch.cuda.is_available():
        print("=== GPU MEMORY ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"GPU {i}: allocated={allocated:.2f} GB reserved={reserved:.2f} GB")


if __name__ == "__main__":
    main()