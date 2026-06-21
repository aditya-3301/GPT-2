#!/usr/bin/env python3
"""
Train a character-level BPE tokenizer on a text file and save the result.

Usage:
    python bpe_train.py train.txt
    python bpe_train.py train.txt --n_merges 2000
    python bpe_train.py train.txt --n_merges 1000 --tokens_out tokens.txt --merges_out merges.txt

Outputs:
    tokens.txt  - one BPE token per line. The line number (0-indexed) IS the
                  token id, so `stoi = {tok: i for i, tok in enumerate(lines)}`
                  reconstructs the exact mapping used during training.
    merges.txt  - one merge rule per line as "left<TAB>right", in the order
                  they were learned. You need this file (not just tokens.txt)
                  to tokenize any new text the same way later.
"""

import argparse
import re
import time
from collections import defaultdict


def split_words_with_sep(text):
    for m in re.finditer(r'(\S+)(\s*)', text):
        word, sep = m.groups()
        if not word:
            continue
        marker = '<para>' if sep.count('\n') >= 2 else '<end>'
        yield word, marker


def get_vocab(text):
    """word -> frequency, where each word is a tuple of symbols (chars + marker)."""
    vocab = defaultdict(int)
    for word, marker in split_words_with_sep(text):
        symbols = tuple(word) + (marker,)
        vocab[symbols] += 1
    return vocab


def get_pair_stats(vocab):
    """pairs: pair -> total freq. pair_to_words: pair -> set of words containing it."""
    pairs = defaultdict(int)
    pair_to_words = defaultdict(set)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
            pair_to_words[pair].add(word)
    return pairs, pair_to_words


def apply_merge_to_word(word, pair):
    a, b = pair
    merged = a + b
    new_word = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(merged)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(text, n_merges):
    vocab = get_vocab(text)
    pairs, pair_to_words = get_pair_stats(vocab)
    merges = []

    t0 = time.time()
    for i in range(n_merges):
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # Only touch words that actually contain best_pair, not the whole vocab.
        affected_words = pair_to_words.pop(best_pair, set())
        pairs.pop(best_pair, None)

        for word in affected_words:
            freq = vocab.pop(word, None)
            if freq is None:
                continue

            for j in range(len(word) - 1):
                p = (word[j], word[j + 1])
                pairs[p] -= freq
                if pairs[p] <= 0:
                    pairs.pop(p, None)
                pair_to_words[p].discard(word)

            new_word = apply_merge_to_word(word, best_pair)
            vocab[new_word] += freq

            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                pairs[p] += freq
                pair_to_words[p].add(new_word)

        if (i + 1) % 100 == 0:
            print(f"  merge {i + 1}/{n_merges} ({time.time() - t0:.1f}s elapsed) -> {best_pair}")

    return vocab, merges


def main():
    ap = argparse.ArgumentParser(description="Train a character-level BPE tokenizer on a text file.")
    ap.add_argument("input", help="Path to input text file (e.g. train.txt)")
    ap.add_argument("--n_merges", type=int, default=1000, help="Number of BPE merges to learn (default: 1000)")
    ap.add_argument("--tokens_out", default="tokens.txt", help="Output path for the vocab list (default: tokens.txt)")
    ap.add_argument("--merges_out", default="merges.txt", help="Output path for the merge rules (default: merges.txt)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Read {len(text):,} characters from {args.input}")
    print(f"Training BPE with up to {args.n_merges} merges...")

    vocab, merges = train_bpe(text, args.n_merges)

    tokens = set()
    for word in vocab:
        tokens.update(word)
    tokens = sorted(tokens)

    with open(args.tokens_out, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

    with open(args.merges_out, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a}\t{b}\n")

    print(f"\nDone. Vocab size: {len(tokens)} | merges learned: {len(merges)}")
    print(f"Tokens saved to: {args.tokens_out}")
    print(f"Merges saved to: {args.merges_out}  (needed later to encode/decode new text)")


if __name__ == "__main__":
    main()