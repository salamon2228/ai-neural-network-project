# AZR Model Trainer v2

### [Русская версия (RU)](README.ru.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Slavikpro557/ai-neural-network-project/blob/main/AZR_Trainer_Colab.ipynb)

A web-based system for creating, training, and analyzing neural language models using the **Absolute Zero Reasoner (AZR)** method with **REINFORCE** self-play for continuous self-improvement.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Features

- **Custom transformer architecture** — create models with any configuration (layers, heads, embedding size)
- **AZR Self-Play training** — the model improves itself through self-play with REINFORCE policy gradient
- **6-component reward system** — diversity, coherence, repetition penalty, length, vocabulary richness, bigram naturalness
- **Dataset catalog** — 110 built-in datasets (Russian & English literature, sci-fi, detective, poetry, code, philosophy, etc.)
- **HuggingFace search** — find and download datasets directly from HuggingFace
- **Multi-dataset training** — attach multiple datasets to a single model
- **Real-time monitoring** — live loss charts, iteration feed, smart analysis of what changed
- **GPU acceleration** — automatic CUDA detection, one-click GPU setup on Windows
- **Google Colab support** — run on a free Tesla T4 GPU with one click
- **Bilingual UI** — full Russian/English interface with language toggle
- **Model comparison** — side-by-side comparison of different models
- **Export** — download trained models as .pt files

## Quick Start

### Option 1: Google Colab (recommended, free GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Slavikpro557/ai-neural-network-project/blob/main/AZR_Trainer_Colab.ipynb)

1. Click the badge above
2. In Colab: **Runtime > Change runtime type > GPU (T4)**
3. Run all cells
4. Click the cloudflare tunnel link to open the web interface

### Option 2: Local installation

```bash
git clone https://github.com/Slavikpro557/ai-neural-network-project.git
cd ai-neural-network-project
pip install -r requirements.txt
python server_with_datasets.py
```

Open http://localhost:8000 in your browser.

### Option 3: Windows one-click

Run `start.bat` — it installs dependencies and starts the server automatically.

For GPU: run `install_gpu.bat` first, then `start_gpu.bat`.

## How It Works

### AZR (Absolute Zero Reasoner) Training

1. **Supervised learning** on text data (books, articles, code) — the model learns language patterns
2. **Self-play generation** — the model generates text samples
3. **Reward evaluation** — a 6-component reward function scores each sample:
   - Token diversity
   - Coherence (bigram probability)
   - Repetition penalty
   - Output length score
   - Vocabulary richness
   - Bigram naturalness
4. **REINFORCE update** — policy gradient adjusts model weights (0.5% RL weight blended with supervised loss)
5. **Validation** — 90/10 train/val split with periodic validation loss logging
6. **Repeat** — the cycle continues, with the model constantly improving

### Architecture

- Custom **Transformer** with configurable depth, width, and attention heads
- **BPE-style tokenizer** trained on your data
- **Checkpointing** every N iterations with resume support
- **DataLoader** with `pin_memory=True` for GPU acceleration

## Dataset Catalog

110 built-in datasets across 16 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Literature (EN) | 22 | Alice in Wonderland, Frankenstein, Pride and Prejudice |
| Literature (RU) | 13 | Anna Karenina, War and Peace, The Idiot, Dead Souls |
| Detective | 4 | Sherlock Holmes, Hound of the Baskervilles |
| Sci-Fi | 9 | Time Machine, War of the Worlds, Frankenstein |
| Poetry | 9 | Shakespeare's Sonnets, Eugene Onegin, Leaves of Grass |
| Philosophy | 8 | The Republic, Meditations, Beyond Good and Evil |
| Fairy Tales | 8 | Grimm's Fairy Tales, Andersen, Arabian Nights |
| + 9 more | 37 | Horror, Adventure, Drama, Code, Science, etc. |

All datasets are public domain or freely licensed. You can also upload your own `.txt`, `.csv`, `.json`, `.jsonl`, `.pdf` files.

## Project Structure

```
ai-neural-network-project/
├── server_with_datasets.py   # Main server (FastAPI)
├── server.py                 # Legacy server
├── model.py                  # Transformer architecture
├── azr_trainer.py            # AZR trainer (base)
├── azr_trainer_resume.py     # AZR trainer with resume support
├── reward_model.py           # 6-component reward system
├── dataset_catalog.py        # 110 built-in datasets
├── dataset_manager.py        # Multi-dataset management
├── tokenizer.py              # BPE tokenizer
├── AZR_Trainer_Colab.ipynb   # Google Colab notebook
├── requirements.txt          # Dependencies
├── start.bat                 # Windows launcher
├── templates/
│   └── index_complete.html   # Web UI (RU/EN)
├── models/                   # Saved models
├── books/                    # Datasets
└── checkpoints/              # Training checkpoints
```

## Model Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| vocab_size | Vocabulary size | 5,000 — 15,000 |
| d_model | Embedding dimension | 128 — 512 |
| num_layers | Transformer layers | 4 — 12 |
| num_heads | Attention heads | 4 — 16 |
| d_ff | Feed-forward size | d_model x 4 |
| max_seq_len | Max sequence length | 128 — 512 |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 4 GB+ RAM (8 GB+ recommended)
- GPU optional (NVIDIA with CUDA for acceleration)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch_size (16 → 8 → 4) or model size |
| Slow training | Install PyTorch with CUDA, reduce max_seq_len |
| Poor generation | Train longer, use more data, increase model size |
| Garbled dataset text | Re-download from catalog (encoding auto-detection fixed) |

## License

Free for personal and commercial use.
