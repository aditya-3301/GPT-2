"""
  tokenisation → data split → model construction → training → generation.

"""

import import_ipynb          
import train                 

from train import (
    # tokeniser
    split_words_with_sep,
    get_vocab,
    get_stats,
    merge_mostfrequent,
    train_bpe,
    encode,
    decode,
    bpe_tokens,
    bpe_vocab_size,
    stoi,
    itos,
    # data
    train as train_data,
    test  as test_data,
    get_batch,
    # model classes
    Head,
    MultiHeadAttention,
    BigramLM,
    # hyperparameters
    block_size,
    batch_size,
    n_embed,
)

import torch
import torch.nn as nn

# ── Hyperparameters
LEARNING_RATE  = 1e-3
MAX_ITERS      = 5000
MAX_NEW_TOKENS = 200
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS_PATH   = 'bigram_weights.pth'


# ── Model 
def build_model():
    m = BigramLM(bpe_vocab_size, n_embed).to(DEVICE)
    total = sum(p.numel() for p in m.parameters())
    return m


# ── Training loop 
def train_model(model, n_iters):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')

    for step in range(n_iters):
        xb, yb  = get_batch(train_data)
        xb, yb  = xb.to(DEVICE), yb.to(DEVICE)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), WEIGHTS_PATH)
    return best_loss


# ── Generation
def generate_text(model, prompt="A scandal", max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(output[0].tolist())


# ── Entry point 
if __name__ == '__main__':

    model = build_model()

    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Loaded existing weights from '{WEIGHTS_PATH}'.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Could not load '{WEIGHTS_PATH}' ({e}). Training from scratch...")
        best_loss = train_model(model, MAX_ITERS)
        print(f"Training done. Best loss: {best_loss:.4f}")

    print("\nGenerated text")
    print(generate_text(model))