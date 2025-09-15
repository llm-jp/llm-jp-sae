# LLM-jp SAE
(Work in progress)

This repository provides code for training and evaluating Sparse Autoencoders (SAEs) on the internal representations of LLM-jp.
We trained SAEs separately on six different checkpoints of [LLM-jp-3-1.8B](https://huggingface.co/llm-jp/llm-jp-3-1.8b) and compared learned features across checkpoints.

- **[Demo Page](https://llm-jp.github.io/llm-jp-sae/)**: Visualize the text samples that activate each SAE feature (100 features per checkpoint).
- **[Model Weights](https://huggingface.co/collections/llm-jp/sparse-autoencoders-67cfcabeaff9c98bb3fdcfb6)**: SAE weights for all six checkpoints.
- **[Paper](https://arxiv.org/abs/2503.06394)**: *"How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders"*


## Usage
### Environment
Python 3.10.12

```
uv init
uv sync
```



```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Need 2GPUs for training.
You should prepare the raw data of en_wiki and ja_wiki in llm-jp-corpus-v3 and checkpoints of LLM-jp-3-1.8B in advance.
Before running the code, please fix the UsrConfig in `config.py` to match your environment.


## Train, Visualize, and Evaluate SAE
### Prepare Data
download llmjp-corpus-v3
```bash
python prepare_data.py
```
### Train SAE
```bash
python train.py
```
### Collect Examples for each Feature
```bash
python collect_examples.py
```
### Evaluate activation patterns
```bash
python evaluate.py
```

## Use Trained SAE and Visualize and Evaluate it
### Prepare Data
download llmjp-corpus-v3
```bash
python prepare_data.py
```
