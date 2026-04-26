import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from pathlib import Path 
from PIL import Image
import base64 
model_id = "mistralai/Ministral-3-14B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
IMAGE_PATH = Path("../../../data/ViLStrUB/images/vp/vp-1-a-i.png") 
#image_url = IMAGE_PATH.as_uri() 

image = Image.open(IMAGE_PATH).convert("RGB") 


def encode_image(image_path): 
    with open(image_path, "rb") as f: 
        return base64.b64encode(f.read()).decode("utf-8") 
    
base64_image = encode_image(IMAGE_PATH) 
image_data_url = f"data:image/png;base64,{base64_image}" 

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "describe the following image ",
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_url},
            },
        ],
    },
]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device="cuda")
#image_sizes = [tokenized["pixel_values"].shape[-2:]]

output = model.generate(
    **tokenized,
    #image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)
