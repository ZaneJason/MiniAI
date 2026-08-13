import json
from pathlib import Path


SPECIAL_TOKENS = ["<unk>"]


class CharTokenizer:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {idx: token for token, idx in stoi.items()}
        self.unk_id = stoi["<unk>"]

    @classmethod
    def build(cls, text: str):
        chars = sorted(set(text))
        stoi = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for ch in chars:
            if ch not in stoi:
                stoi[ch] = len(stoi)
        return cls(stoi)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["stoi"])

    def save(self, path):
        Path(path).write_text(
            json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        pieces = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            pieces.append("�" if token == "<unk>" else token)
        return "".join(pieces)
