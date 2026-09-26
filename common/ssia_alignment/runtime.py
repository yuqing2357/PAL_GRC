"""Retained CUDA/DDP/EMA runtime optimizations from the prior END trainer."""
from __future__ import annotations

import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

GPU_BATCH_KEYS = {"z", "mask", "q_of_p", "valid_pair"}


def ddp_setup() -> tuple[int, int, int, bool]:
    if "RANK" not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training")
        torch.cuda.set_device(0)
        return 0, 1, 0, False
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return dist.get_rank(), dist.get_world_size(), local_rank, True


def barrier(active: bool) -> None:
    if active:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def seed(seed: int, rank: int) -> None:
    value = int(seed) + 100_003 * int(rank)
    random.seed(value); np.random.seed(value % (2**32 - 1)); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def rng_state(local_rank: int) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(local_rank)}


def restore_rng_state(state: dict, local_rank: int) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"].cpu()); torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device=local_rank)


def collect_rng_states(local_rank: int, rank: int, world_size: int) -> list[dict] | None:
    local = rng_state(local_rank)
    if world_size == 1:
        return [local]
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.parameter_names = {name for name, _ in model.named_parameters()}
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for name, value in model.state_dict().items():
            source = value.detach()
            if name in self.parameter_names and torch.is_floating_point(source):
                self.state[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                self.state[name].copy_(source)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.state = {name: value.detach().clone() for name, value in state.items()}


def build_scheduler(cfg, optimizer: torch.optim.Optimizer, max_steps: int):
    warmup, minimum = int(cfg.train.get("warmup_steps", 0)), float(cfg.train.get("min_lr_ratio", 0.1))
    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return max(1.0e-8, step / max(1, warmup))
        progress = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    kwargs = {"lr": float(cfg.train.lr), "weight_decay": float(cfg.train.get("weight_decay", 0.0)), "betas": tuple(float(x) for x in cfg.train.get("betas", [0.9, 0.999]))}
    if bool(cfg.train.get("fused_optimizer", True)):
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


class CUDAPrefetcher:
    def __init__(self, loader, device: torch.device) -> None:
        self.iterator, self.device = iter(loader), device
        self.stream, self.next_batch = torch.cuda.Stream(device=device), None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None; return
        with torch.cuda.stream(self.stream):
            self.next_batch = {key: value.to(self.device, non_blocking=True) if key in GPU_BATCH_KEYS and torch.is_tensor(value) else value for key, value in batch.items()}

    def next(self) -> dict:
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        for key in GPU_BATCH_KEYS:
            if torch.is_tensor(batch.get(key)) and batch[key].is_cuda:
                batch[key].record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch


def maybe_warm_page_cache(cfg, data_root: Path, *, rank: int, distributed: bool) -> None:
    if bool(cfg.data.get("warm_page_cache", True)) and rank == 0:
        if str(cfg.data.get("format", "")).lower() == "prepared_imp_rgt_v1":
            prepared = data_root / "train" if (data_root / "train").is_dir() else data_root / "prepared" / "train"
            files = [prepared / "imp.f16.npy", prepared / "rgt.f16.npy"]
        else:
            files = [
                data_root.parent / "preprocessed_p99_per_curve_f32_v1" / "train" / "z.f32.npy",
                data_root / "train" / "rgt.f16.npy",
            ]
        for path in files:
            with path.open("rb", buffering=0) as handle:
                while handle.read(64 * 1024 * 1024):
                    pass
        print("page-cache warmup completed for training input and RGT labels", flush=True)
    barrier(distributed)
