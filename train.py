import argparse
import json
import os
from contextlib import nullcontext

import torch
from config import TrainConfig, return_save_dir
from datasets import load_dataset
from model import TopKSAE, TopKSAEConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup
from utils.activation_buffer import ActivationBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        distances = torch.norm(points - guess, dim=1)
        # Avoid division by zero
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def train(sae, activation_buffer, train_cfg):
    optimizer = torch.optim.Adam(sae.parameters(), lr=train_cfg.lr, eps=6.25e-10)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.num_warmup_steps)
    autocast_context = (
        nullcontext() if device.type == "cpu" else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )
    total_tokens = train_cfg.tokens
    logging_step = train_cfg.logging_step
    n_token = 0
    train_loss = []
    loss_accum = 0.0

    sae.train()
    pbar = tqdm(total=total_tokens)

    for step, x in enumerate(activation_buffer):
        x = x / x.norm(dim=-1, keepdim=True)
        if step == 0:
            median = geometric_median(x)
            sae.b_dec.data = median.to(sae.dtype)

        with autocast_context:
            x_hat = sae(x)
            e = x - x_hat
            loss = e.pow(2).sum(dim=-1).mean()
            loss.backward()
            loss_accum += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            sae.set_decoder_norm_to_unit_norm()

        n_token += x.shape[0]
        pbar.update(x.shape[0])
        if (step + 1) % logging_step == 0:
            train_loss.append(loss_accum / logging_step)
            loss_accum = 0.0

        if n_token >= total_tokens:
            break

    return sae, train_loss


def evaluate(sae, activation_buffer, train_cfg):
    autocast_context = (
        nullcontext() if device.type == "cpu" else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    )
    total_tokens = train_cfg.tokens_eval
    n_token = 0
    loss_accum = 0.0
    cos_sim = 0.0

    sae.eval()
    pbar = tqdm(total=total_tokens)

    with torch.no_grad():
        for step, x in enumerate(activation_buffer):
            x = x / x.norm(dim=-1, keepdim=True)

            with autocast_context:
                x_hat = sae(x)
                e = x - x_hat
                loss = e.pow(2).sum(dim=-1).mean()
                loss_accum += loss.item()

                cos_sim += torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

            n_token += x.shape[0]
            pbar.update(x.shape[0])

            if n_token >= total_tokens:
                break

    return loss_accum / (step + 1), cos_sim / (step + 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./output")

    parser.add_argument("--local_model", type=str)
    parser.add_argument("--hf_model", type=str)
    parser.add_argument("--hf_revision", type=str, default="main")
    parser.add_argument("--layer", type=int, default=12, help="Layer (>= 1) to analyze.")

    parser.add_argument("--local_data", type=str, help="Path to the directory containing train_data.pt and val_data.pt")
    parser.add_argument("--hf_data", type=str, help="HuggingFace dataset repository")
    parser.add_argument("--hf_name", type=str)

    parser.add_argument("--num_latents", type=int, default=32768, help="Dimensionality of SAE's hidden layer")
    parser.add_argument("--k", type=int, default=32, help="K parameter for SAE (sparsity)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    ### save dir for this experiment
    save_dir = return_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)

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
    train_cfg = TrainConfig(lr=args.lr)
    if args.hf_data:
        dataset = load_dataset(args.hf_data, name=args.hf_name, split="train", streaming=True)
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, padding=True, max_length=train_cfg.ctx_len),
            batched=True,
        )

    ### Prepare Activation Buffer
    activation_buffer = ActivationBuffer(
        model=model,
        submodule=submodule,
        dataset=dataset,
        buffer_size=train_cfg.buffer_size,
        refresh_batch_size=train_cfg.bs_lm,
        batch_size=train_cfg.bs_sae,
        device=device,
    )

    ### Initialize SAE
    sae_cfg = TopKSAEConfig(num_latents=args.num_latents, k=args.k)
    sae = TopKSAE(sae_cfg).to(device)

    ### Train SAE
    sae, train_loss = train(
        sae=sae,
        activation_buffer=activation_buffer,
        train_cfg=train_cfg,
    )

    ### Save SAE
    sae.save_pretrained(save_dir)

    ### Evaluate SAE
    eval_loss, cos_sim = evaluate(sae=sae, activation_buffer=activation_buffer, train_cfg=train_cfg)

    print(f"Eval loss: {eval_loss}")
    print(f"Cosine similarity: {cos_sim}")
    with open(f"{save_dir}/results.json", "w") as f:
        json.dump({"eval_loss": eval_loss, "cosine_similarity": cos_sim, "train_loss": train_loss}, f, indent=4)


if __name__ == "__main__":
    main()
