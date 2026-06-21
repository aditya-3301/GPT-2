# GPT-2 (mini) — Sherlock Holmes char/BPE language model

A small GPT-style transformer (2 blocks, 4 heads, 256-dim embeddings) trained
on Sherlock Holmes text using a custom BPE tokenizer.

## Files

| File | Purpose |
|---|---|
| `model.py` | Tokenizer (BPE encode/decode) + model architecture (`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `BigramLM`). No training code. |
| `main.py` | Loads `bigram_weights.pth` and generates text. **No training.** |
| `train.ipynb` | Full training notebook — data loading, tokenizer setup, training loop, loss curves. |
| `tokens.txt` | BPE vocabulary (one token per line). |
| `merges.txt` | BPE merge rules, in priority order. |
| `bigram_weights.pth` | Trained model weights (best checkpoint by val loss). |
| `train.txt` | Training corpus. |

## Running the model (generate text)

No training needed — just load the finalized weights.

```bash
pip install torch
python main.py
```

Run this from inside the folder containing `tokens.txt`, `merges.txt`, and
`bigram_weights.pth` (they're loaded via relative paths).

To change the prompt or generation length, edit the bottom of `main.py`:

```python
print(generate_text(model, prompt="The detective frowned", max_new_tokens=300))
```

## Training from scratch

Training happens in `train.ipynb`, not `main.py`. Open the notebook and run
all cells top to bottom:

1. Loads `train.txt`, builds/loads the BPE tokenizer (`tokens.txt` + `merges.txt`)
2. Splits encoded tokens 90/10 into train/val
3. Builds the model and runs the training loop
4. Saves the best checkpoint (lowest val loss) to `bigram_weights.pth`

If you change any hyperparameter below, you must also update the matching
value in `model.py` so `main.py` can correctly reconstruct the model when
loading the saved weights.

## Main hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `block_size` | 256 | context length (tokens) |
| `batch_size` | 16 | sequences per training step |
| `n_embed` | 256 | embedding dimension |
| `n_head` | 4 | attention heads per block |
| `n_layer` | 2 | number of transformer blocks |
| `dropout` | 0.2 | applied in attention + feedforward |
| `bpe_vocab_size` | ~1103 | determined by `tokens.txt`, not hand-set |
| `max_lr` | 1e-3 | peak learning rate |
| `min_lr` | 1e-4 | cosine decay floor |
| `max_steps` | 25000 | total training steps (cosine schedule length) |

## Notes

- Best checkpoint is selected by **val loss**, not train loss, and is saved
  automatically during training (`bigram_weights.pth` is overwritten whenever
  val loss improves).
- The tokenizer in `model.py` caches BPE merges per unique word
  (`lru_cache`), so re-encoding repeated words (common in natural text) is
  fast even on a CPU.

## Directory structure

```
GPT-2/
├── main.py               # generate text (no training)
├── model.py              # tokenizer + model architecture
├── train.ipynb           # training notebook
├── train.txt             # training corpus
├── tokens.txt            # BPE vocabulary
├── merges.txt            # BPE merge rules
├── bigram_weights.pth    # trained weights
└── README.md
```

stuff required for **generating** (`python main.py`): `main.py`, `model.py`,
`tokens.txt`, `merges.txt`, `bigram_weights.pth`.

stuff Required for **training** (`train.ipynb`): `train.ipynb`, `train.txt`,
`tokens.txt`, `merges.txt`.