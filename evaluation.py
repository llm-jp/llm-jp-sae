import argparse
import os
from contextlib import nullcontext

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from utils.nethook import Trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EVAL_TOKENS = 1_000_000
CONTEXT_LENGTH = 1024


def evaluate(sae, model, submodule, dataloader, total_tokens, remove_bos=True):
    autocast_context = (
        nullcontext() if device.type == "cpu" else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )
    n_token = 0

    sae.eval()
    pbar = tqdm(total=total_tokens)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = torch.tensor(batch["input_ids"]).to(device)  # B x T
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)  # B x T
            if remove_bos:
                left_most_indices = (attention_mask == 1).float().cumsum(dim=-1)
                attention_mask[left_most_indices == 1] = 0

            with Trace(model, submodule, stop=True) as trace:
                _ = model(input_ids)
            x_BTD = trace.output[0]
            x_BTD = x_BTD / x_BTD.norm(dim=-1, keepdim=True)

            with autocast_context:
                x_hat_BTD, topk_acts_BTK, topk_indices_BTK = sae(x_BTD, output_features=True)

            # x_BTD.shape = torch.Size([4, 1024, 2048])
            # x_hat_BTD.shape = torch.Size([4, 1024, 2048])
            # topk_acts_BTK.shape = torch.Size([4, 1024, 64])
            # topk_indices_BTK.shape = torch.Size([4, 1024, 64])
            # attention_mask.shape = torch.Size([4, 1024])

            ### Write Evaluation Code

            n_token_ = attention_mask.sum().item()
            n_token += n_token_
            pbar.update(n_token_)

            if n_token >= total_tokens:
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./output")

    parser.add_argument("--local_model", type=str)
    parser.add_argument("--hf_model", type=str)
    parser.add_argument("--hf_revision", type=str, default="main")
    parser.add_argument("--layer", type=int, default=12, help="Layer (>= 1) to analyze.")

    parser.add_argument("--hf_sae", type=str)

    parser.add_argument("--local_data", type=str, help="Path to the directory containing train_data.pt and val_data.pt")
    parser.add_argument("--hf_data", type=str, help="HuggingFace dataset repository")
    parser.add_argument("--hf_name", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    ### save dir for this experiment
    os.makedirs(args.save_dir, exist_ok=True)

    ### Load Language Model
    print("Loading language model...")
    if args.local_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.local_model,
            torch_dtype="bfloat16",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.local_model)
    elif args.hf_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            revision=args.hf_revision,
            trust_remote_code=True,
            torch_dtype="bfloat16",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, revision=args.hf_revision)
    else:
        raise ValueError("Specify either --local_model or --hf_model")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ### Get submodule
    if 1 <= args.layer <= len(model.model.layers):
        submodule = f"model.layers.{args.layer - 1}"
    else:
        raise ValueError(f"Layer {args.layer} is out of range.")

    ### Load Data
    if args.hf_data:
        dataset = load_dataset(args.hf_data, name=args.hf_name, split="train", streaming=True)
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=CONTEXT_LENGTH),
            batched=True,
        )
    dataloader = iter(dataset.iter(batch_size=BATCH_SIZE))

    ### Initialize SAE
    sae = AutoModel.from_pretrained(
        args.hf_sae,
        trust_remote_code=True,
    ).to(device)

    ### Evaluate SAE
    eval_loss, cos_sim = evaluate(
        sae=sae, model=model, submodule=submodule, dataloader=dataloader, total_tokens=EVAL_TOKENS
    )


if __name__ == "__main__":
    main()
