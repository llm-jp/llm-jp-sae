import gc

import torch
from nethook import Trace


class ActivationBuffer:
    def __init__(
        self,
        model,
        submodule,
        dataset,
        buffer_size: int,
        refresh_batch_size: int,
        batch_size: int,
        device: str = "cpu",
        remove_bos: bool = True,
    ):
        self.d_submodule = model.config.hidden_size
        self.activations = torch.empty(0, self.d_submodule, device=device)
        self.read = torch.zeros(0).bool()

        self.dataset = dataset
        self.model = model
        self.submodule = submodule
        self.buffer_size = buffer_size
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = batch_size
        self.device = device
        self.remove_bos = remove_bos

        self._iterator = iter(self.dataset.iter(batch_size=refresh_batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        with torch.no_grad():
            # if buffer is less than half full, refresh
            # if (~self.read).sum() < self.activation_buffer_size // 2:
            #     self.refresh()
            if (~self.read).sum() < self.refresh_batch_size:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[torch.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]

    def tokenized_batch(self):
        batch = next(self._iterator)
        # return {k: torch.tensor(v).to(self.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        return torch.tensor(batch["input_ids"]).to(self.device), torch.tensor(batch["attention_mask"]).to(self.device)

    def refresh(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = torch.empty(
            self.buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.model.dtype,
        )

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        while current_idx < self.buffer_size:
            with torch.no_grad():
                input_ids, attention_mask = self.tokenized_batch()
                with Trace(self.model, self.submodule) as trace:
                    _ = self.model(input_ids)
                    hidden_states = trace.output[0]  # B x T x D

            if self.remove_bos:
                ### for left padding
                left_most_indices = (attention_mask == 1).float().cumsum(dim=-1)
                attention_mask[left_most_indices == 1] = 0
            hidden_states = hidden_states[attention_mask != 0]

            remaining_space = self.buffer_size - current_idx
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states
            current_idx += len(hidden_states)

        self.read = torch.zeros(len(self.activations), dtype=torch.bool, device=self.device)
