"""
main.py loads the finalized BigramLM weights and generate text.

No training. Imports the tokenizer + model from model.py
"""

import torch

from model import (
    BigramLM,
    encode,
    decode,
    bpe_vocab_size,
    n_embed,
    DEVICE,
)

WEIGHTS_PATH   = 'bigram_weights.pth'
MAX_NEW_TOKENS = 200


def load_model():
    model = BigramLM(bpe_vocab_size, n_embed).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model


def generate_text(model, prompt="Holmes was", max_new_tokens=MAX_NEW_TOKENS):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(output[0].tolist())


if __name__ == '__main__':
    model = load_model()
    print(f"Loaded weights from '{WEIGHTS_PATH}' on {DEVICE}.\n")

    print("Generated text:\n")
    print(generate_text(model))