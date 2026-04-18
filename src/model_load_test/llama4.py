from __future__ import annotations

import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Llama4ForConditionalGeneration

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def print_gpu_info() -> None:
    print("=== CUDA INFO ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name}, total_memory={total_gb:.2f} GB")

    print()


def load_model_and_processor():
    print("=== LOADING PROCESSOR ===")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Processor loaded.\n")

    attempts = [
        {
            "name": "float16 + default attention",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            },
        },
        {
            "name": "bfloat16 + default attention",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            },
        },
        {
            "name": "float16 + flex_attention",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "attn_implementation": "flex_attention",
            },
        },
    ]

    last_error = None

    for attempt in attempts:
        print(f"=== TRYING MODEL LOAD: {attempt['name']} ===")
        try:
            model = Llama4ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                **attempt["kwargs"],
            )
            print(f"SUCCESS: {attempt['name']}")
            print(f"hf_device_map: {model.hf_device_map}\n")
            return processor, model, attempt["name"]
        except Exception as e:
            last_error = e
            print(f"FAILED: {attempt['name']}")
            print(f"{type(e).__name__}: {e}\n")

    raise RuntimeError(f"All load attempts failed. Last error: {last_error}")


def build_test_input(processor, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "What is in this image? Answer in one short sentence."
                },
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return inputs


def move_inputs_to_model_device(inputs, model):
    inputs = inputs.to(model.device)
    
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(
            device=model.device,
            dtype=torch.float16, 
        )

    return inputs 

def run_generation(processor, model, image: Image.Image) -> None:
    print("=== BUILDING INPUTS ===")
    inputs = build_test_input(processor, image)
    inputs = move_inputs_to_model_device(inputs, model)

    print("Input keys:", list(inputs.keys()))
    print("input_ids shape:", tuple(inputs["input_ids"].shape))
    print()

    print("=== RUNNING GENERATION ===")
    with torch.inference_mode():
        
        if "pixel_values" in inputs:
            print("pixel_values dtype:", inputs["pixel_values"].dtype)
        print("model dtype:", getattr(model, "dtype", "unknown")) 
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    print("=== MODEL RESPONSE ===")
    print(response)
    print()


def print_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        return

    print("=== GPU MEMORY SNAPSHOT ===")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"GPU {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
    print()


def main() -> None:
    print_gpu_info()

    # Use a local image file if you prefer. This avoids internet dependency.
    # Replace this with one of your own dataset images.
    image_path = Path("../../../data/ViLStrUB/images/vp/vp-1-a-i.png") 

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}. Replace with any local PNG/JPG."
        )

    image = Image.open(image_path).convert("RGB")

    try:
        processor, model, load_mode = load_model_and_processor()
        print(f"Chosen load mode: {load_mode}\n")

        print_memory_snapshot()
        run_generation(processor, model, image)
        print_memory_snapshot()

        print("=== SUCCESS ===")
        print("Llama-4 Scout loaded and generated successfully on this node.")

    except Exception as e:
        print("=== FAILURE ===")
        print(f"{type(e).__name__}: {e}")
        print()
        traceback.print_exc()


if __name__ == "__main__":
    main()