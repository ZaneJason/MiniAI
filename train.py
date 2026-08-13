import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import ModelConfig, TrainConfig
from model import MiniGPT
from tokenizer import CharTokenizer


def load_tokens(path, dtype):
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size + 1:
        raise ValueError("Dataset is too small for the configured block_size")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(np.array(data[i : i + block_size], dtype=np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(np.array(data[i + 1 : i + 1 + block_size], dtype=np.int64)) for i in ix]
    )
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device, amp_dtype):
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, model.config.block_size, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def main():
    train_cfg = TrainConfig()
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"device: {gpu_name}")
    else:
        print("device: CPU (training will be slow)")

    data_dir = Path(train_cfg.data_dir)
    tokenizer = CharTokenizer.load(data_dir / "tokenizer.json")
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    np_dtype = np.dtype(meta["dtype"])
    train_data = load_tokens(data_dir / "train.bin", np_dtype)
    val_data = load_tokens(data_dir / "val.bin", np_dtype)

    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = MiniGPT(model_cfg).to(device)
    print(f"parameters: {model.num_parameters() / 1e6:.2f}M")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=device.type == "cuda",
    )

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    started = time.time()

    for step in range(train_cfg.max_steps):
        lr = get_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg.gradient_accumulation_steps):
            x, y = get_batch(train_data, train_cfg.batch_size, model_cfg.block_size, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / train_cfg.gradient_accumulation_steps
            loss_accum += loss.detach().item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        current = step + 1
        if current % train_cfg.log_interval == 0:
            elapsed = time.time() - started
            print(f"step {current:06d} | loss {loss_accum:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            started = time.time()

        if current % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, train_cfg, device, amp_dtype)
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")

        if current % train_cfg.save_interval == 0 or current == train_cfg.max_steps:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
                "step": current,
            }
            numbered = out_dir / f"step_{current:08d}.pt"
            torch.save(checkpoint, numbered)
            torch.save(checkpoint, out_dir / "latest.pt")
            print(f"saved: {numbered}")


if __name__ == "__main__":
    main()
