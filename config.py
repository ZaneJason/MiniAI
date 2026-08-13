from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 256
    vocab_size: int = 0  # prepare_data.py / tokenizer determines this
    n_layer: int = 10
    n_head: int = 10
    n_embd: int = 640
    dropout: float = 0.0
    bias: bool = False


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "checkpoints"
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 50_000
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 1_000
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337
