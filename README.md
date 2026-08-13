# MiniAI

一个从**随机权重**开始训练的小型中文 GPT 项目。它不下载 Qwen/Llama 等现成模型权重，也不调用外部大模型 API；核心 Transformer、训练循环、Tokenizer 与生成逻辑都在本仓库中实现。

## 目标

第一版默认配置约为 50M 级参数（实际参数量取决于语料字符表大小），适合单卡 GPU 学习和实验。默认使用字符级 Tokenizer，对中文非常直观：先从你自己的 `data/train.txt` 建立词表，再将语料编码为二进制 token 数据，最后从随机初始化参数开始自回归预训练。

## 项目结构

```text
MiniAI/
├── config.py
├── model.py
├── tokenizer.py
├── prepare_data.py
├── train.py
├── generate.py
├── requirements.txt
└── data/
    └── README.md
```

## 1. 环境

建议 Python 3.11+ 与支持 CUDA 的 PyTorch。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> `requirements.txt` 不锁死 PyTorch 的 CUDA wheel。云 GPU 环境通常已经预装 PyTorch；如果没有，请按你的 CUDA 环境安装官方 PyTorch CUDA 版本。

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. 准备自己的训练数据

把 UTF-8 纯文本放到：

```text
data/train.txt
```

第一轮为了验证训练链路，几十 MB 就能跑；想让模型真正学到更稳定的中文语言规律，需要明显更大的、高质量且有合法使用权的语料。

## 3. 建词表并编码语料

```bash
python prepare_data.py
```

会生成：

```text
data/tokenizer.json
data/train.bin
data/val.bin
```

Tokenizer 是直接从你的训练语料构建的字符级 Tokenizer，并非使用第三方模型词表。

## 4. 开始训练

```bash
python train.py
```

训练过程中会输出 loss、学习率、吞吐量，并周期性保存 checkpoint：

```text
checkpoints/step_00001000.pt
checkpoints/latest.pt
```

默认支持 CUDA AMP、梯度累积、梯度裁剪、warmup + cosine decay。

## 5. 生成文本

```bash
python generate.py --prompt "人工智能" --max-new-tokens 200
```

也可以指定 checkpoint：

```bash
python generate.py --checkpoint checkpoints/latest.pt --prompt "生命科学"
```

## 默认模型

默认主要配置：

- `n_layer = 10`
- `n_head = 10`
- `n_embd = 640`
- `block_size = 256`
- tied token embedding / LM head
- Pre-LN Transformer
- GELU MLP
- causal self-attention

实际参数量会在启动训练时打印。

## 重要说明

这是一个教育与实验项目。50M 左右的模型可以让你完整体验“从随机参数开始学习语言”的过程，但不要期待它达到商业大模型的知识量、推理能力或对话能力。后续可以逐步扩展到 BPE/SentencePiece、RoPE、RMSNorm、SwiGLU、Flash Attention、分布式训练、指令微调等。

## License

代码可按仓库 License 使用；训练数据的许可由数据提供方决定，请只使用你有权使用的语料。
