from pathlib import Path

import numpy as np

from tokenizer import CharTokenizer


DATA_DIR = Path("data")
SOURCE = DATA_DIR / "train.txt"


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            "Missing data/train.txt. Put your UTF-8 training corpus there first."
        )

    text = SOURCE.read_text(encoding="utf-8")
    if len(text) < 10_000:
        print("Warning: corpus is very small; this is only suitable for a smoke test.")

    tokenizer = CharTokenizer.build(text)
    tokenizer.save(DATA_DIR / "tokenizer.json")

    ids = tokenizer.encode(text)
    if tokenizer.vocab_size >= 65535:
        dtype = np.uint32
    else:
        dtype = np.uint16

    split = int(len(ids) * 0.98)
    train_ids = np.asarray(ids[:split], dtype=dtype)
    val_ids = np.asarray(ids[split:], dtype=dtype)
    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")

    (DATA_DIR / "meta.json").write_text(
        '{"dtype": "%s", "vocab_size": %d}' % (np.dtype(dtype).name, tokenizer.vocab_size),
        encoding="utf-8",
    )

    print(f"characters: {len(text):,}")
    print(f"vocab size: {tokenizer.vocab_size:,}")
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens: {len(val_ids):,}")


if __name__ == "__main__":
    main()
