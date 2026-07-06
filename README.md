# LaViSA (Language and Vision Structural Ambiguity) Benchmark
Code repository for the ARR May Submission of **LaViSA**, a benchmark for evaluating current Vision-Language Models (VLMs) on processing structural ambiguity using visual information. 

Given an ambiguous sentence and a clarifying image, a model is asked to identify the intended interpretation. LaViSA covers seven types of structural ambiguity and is designed to analyse both model-level performance and category-specific failure patterns.
---

# Dataset  
The dataset is available on [Hugging Face](https://huggingface.co/datasets/Kerimu/LaViSA/tree/main): 

After downloading the dataset, place it in your preferred directory and set the following paths in:


`./src/main_eval/const.py`

- `json_path` 
    - Path to the benchmark JSON file 
- `image_dir`
    - Path to the benchmark iamge directory 

# Environment 
All experiments were conducted under `venv` with Python 3.12.2. 

Install the required packages according to the `requirements.txt` file: 

pip install -r requirements.txt 

# Experiments 
Run the code as follows:

./run.sh MODEL_TYPE MODEL_CARD

The model types used in our research are:
`["gemini", "openai", "Sqwen", "gemma", "llava"]`

Below are the model cards used in our experiments: 

```md id="7xg4hp"
| VLM | Model Card |
|---|---|
| GPT-5.2 | `gpt-5.2` |
| Gemini 3.1 Pro | `gemini-3.1-pro-preview` |
| Gemini 3.1 Flash-Lite | `gemini-3-1-flash-lite` |
| LLaVA-OneVision-1.5-8B-Instruct | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` |
| Qwen3-VL-4B-Instruct | `Qwen/Qwen3-VL-4B-Instruct` |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` |
| Qwen3-VL-32B-Instruct | `Qwen/Qwen3-VL-32B-Instruct` |
| Qwen3-VL-32B-Thinking | `Qwen/Qwen3-VL-32B-Thinking` |
| Gemma3-4b-it | `google/gemma-3-4b-it` |
| Gemma3-12b-it | `google/gemma-3-12b-it` |
| Gemma3-27b-it | `google/gemma-3-27b-it` |
```

# Citation 

If you use LaViSA in your research, please cite: 

@misc{lee2026lavisa,
  title={LaViSA: A Language and Vision Structural Ambiguity Benchmark},
  author={Lee, Sangmyeong and Inadumi, Shun and Yoshino, Koichiro},
  year={2026},
  eprint={2606.19552},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

# License 

This project is released under the CC BY-SA 4.0 License. 
