Before we begin, let's set up the machine for Inference Engineering. I am using an Apple M4 Mac Mini with 16GB unified memory. 

```bash
conda deactivate # if conda is running
uv init --bare --python 3.12
uv add mlx mlx-lm transformers torch huggingface_hub

```
When done installing the dependencies, run:
```bash
uv run hf auth login
```
Paste a read token you get from huggingface.co > Settings > Access Tokens

Next, we download the models we'll use for this lesson, namely:
1. Qwen 3-1.7B
2. HuggingFaceTB/SmolLM2-135M

```bash
uv run hf download Qwen/Qwen3-1.7B
uv run hf download HuggingFaceTB/SmolLM2-135M
```

