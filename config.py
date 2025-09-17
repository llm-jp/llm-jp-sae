from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_warmup_steps: int = 1000
    ctx_len: int = 1024
    inf_bs_expansion: int = 2
    logging_step: int = 50
    lr: float = 1e-3
    bs_lm: int = 32
    bs_sae: int = 32768
    num_workers: int = 12
    buffer_size: int = 3_000_000
    tokens: int = 100_000_000
    tokens_eval: int = 10_000_000


def return_save_dir(args):
    if args.local_model:
        model = args.local_model.replace("/", "_")
    elif args.hf_model:
        model = f"{args.hf_model.replace('/', '_')}_{args.hf_revision}"

    save_dir = f"{args.save_dir}/{model}/layer{args.layer}/n{args.num_latents}_k{args.k}/lr{args.lr}"
    return save_dir
