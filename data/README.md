# Training data

将你有权使用的 UTF-8 纯文本训练语料保存为：

```text
data/train.txt
```

然后在项目根目录执行：

```bash
python prepare_data.py
```

脚本会基于这份语料从零构建字符级词表，并生成 `train.bin`、`val.bin`、`tokenizer.json` 和 `meta.json`。这些训练产物默认不会提交到 GitHub。

建议先用小语料完成 smoke test，确认 loss 能正常下降和 checkpoint 能正常生成后，再切换到大规模语料训练。
