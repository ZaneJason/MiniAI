import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import MiniGPT
from tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", default="人工智能")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer.load(Path("data") / "tokenizer.json")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt cannot be empty")
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
