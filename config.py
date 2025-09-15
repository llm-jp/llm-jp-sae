from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

@dataclass
class DataConfig:
    seq_len: int = 64
    token_num: int = 165_000_000
    data_pths: dict = field(
        default_factory=lambda: {
            "ja_wiki": [f"ja/ja_wiki/train_{str(i)}.jsonl" for i in range(14)],
            "en_wiki": [f"en/en_wiki/train_{str(i)}.jsonl" for i in range(67)],
        }
    )
    batch_size_tokenizer: int = 5000
    train_val_test_ratio: list = field(default_factory=lambda: [0.8, 0.1, 0.1])


class SaeConfig(PretrainedConfig):
    def __init__(
        self,
        d_in: int = 2048,
        n_dim: int = 32768,
        k: int = 32,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs, torch_dtype=torch_dtype)
        self.d_in = d_in
        self.n_dim = n_dim
        self.k = k


@dataclass
class TrainConfig:
    num_warmup_steps: int = 1000
    batch_size: int = 512
    ctx_len: int = 1024
    inf_bs_expansion: int = 2
    logging_step: int = 50
    lr: float = 1e-3
    bs_lm: int = 32
    bs_sae: int = 32768
    num_workers: int = 12
    buffer_size: int = 500_000
    tokens: int = 100_000_000
    tokens_eval: int = 10_000_000


@dataclass
class EvalConfig:
    num_examples: int = 50
    act_threshold_p: float = 0.7


def return_save_dir(args):
    if args.local_model:
        model = args.local_model.replace("/", "_")
    elif args.hf_model:
        model = f"{args.hf_model.replace('/', '_')}_{args.hf_revision}"

    save_dir = f"{args.save_dir}/{model}/layer{args.layer}/n{args.num_latents}_k{args.k}/lr{args.lr}"
    return save_dir
