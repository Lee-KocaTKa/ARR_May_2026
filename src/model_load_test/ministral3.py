from __future__ import annotations

import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512"


def print_gpu_info() -> None:
    print("=== CUDA INFO ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"GPU {i}: {props.name}, total_memory={total_gb:.2f} GB")
    print()


def load_model_and_tokenizer():
    print("=== LOADING TOKENIZER ===")
    tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID)
    print("Tokenizer loaded.\n")

    attempts = [
        {
            "name": "float16 + device_map=auto",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
            },
        },
        {
            "name": "default dtype + device_map=auto",
            "kwargs": {
                "device_map": "auto",
            },
        },
        {
            "name": "bfloat16 + device_map=auto",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            },
        },
    ]

    last_error = None

    for attempt in attempts:
        print(f"=== TRYING MODEL LOAD: {attempt['name']} ===")
        try:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                **attempt["kwargs"],
            )
            print(f"SUCCESS: {attempt['name']}")
            if hasattr(model, "hf_device_map"):
                print("hf_device_map:", model.hf_device_map)
            print()
            return tokenizer, model, attempt["name"]
        except Exception as e:
            last_error = e
            print(f"FAILED: {attempt['name']}")
            print(f"{type(e).__name__}: {e}\n")

    raise RuntimeError(f"All load attempts failed. Last error: {last_error}")


def move_batch_to_cuda(batch: dict) -> dict:
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            if key == "pixel_values":
                moved[key] = value.to(device="cuda", dtype=torch.float16)
            else:
                moved[key] = value.to(device="cuda")
        else:
            moved[key] = value

    return moved


def main() -> None:
    print_gpu_info()

    image_path = Path("../../../data/ViLStrUB/images/vp/vp-1-a-i.png")
    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}. Put one local image there first."
        )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in one short sentence.",
                },
                {
                    "type": "image",
                    "image": image,
                },
            ],
        },
    ]

    try:
        tokenizer, model, load_mode = load_model_and_tokenizer()
        print(f"Chosen load mode: {load_mode}\n")

        tokenized = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )

        print("=== TOKENIZED INPUT ===")
        for k, v in tokenized.items():
            if torch.is_tensor(v):
                print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"{k}: {type(v)}")
        print()

        tokenized = move_batch_to_cuda(tokenized)

        image_sizes = None
        if "pixel_values" in tokenized:
            image_sizes = [tokenized["pixel_values"].shape[-2:]]

        print("=== GENERATING ===")
        with torch.inference_mode():
            output = model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=64,
                do_sample=False,
            )[0]

        prompt_len = len(tokenized["input_ids"][0])
        decoded_output = tokenizer.decode(
            output[prompt_len:],
            skip_special_tokens=True,
        )

        print("=== MODEL RESPONSE ===")
        print(decoded_output)
        print("\n=== SUCCESS ===")

    except Exception as e:
        print("=== FAILURE ===")
        print(f"{type(e).__name__}: {e}")
        print()
        traceback.print_exc()


if __name__ == "__main__":
    main()