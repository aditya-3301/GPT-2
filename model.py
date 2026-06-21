"""
model.py — standalone inference-only module.

Contains only what's needed to load the trained weights and generate text:
tokenizer (BPE encode/decode using pretrained merges.txt + tokens.txt),
model architecture (Head, MultiHeadAttention, FeedForward, Block, BigramLM),
and the hyperparameters used at training time.

No training loop
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters (must match training exactly so the saved weights load) ──
block_size = 256
n_embed    = 256
dropout    = 0.2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Tokenizer ─────────────────────────────────────────────────────────────
def split_words_with_sep(text):
    for m in re.finditer(r'(\S+)(\s*)', text):
        word, sep = m.groups()
        if not word:
            continue
        marker = '<para>' if sep.count('\n') >= 2 else '<end>'
        yield word, marker


with open('tokens.txt', encoding='utf-8') as f:
    bpe_tokens = [line.rstrip('\n') for line in f]

with open('merges.txt', encoding='utf-8') as f:
    merges = [tuple(line.rstrip('\n').split('\t')) for line in f]

stoi = {token: i for i, token in enumerate(bpe_tokens)}
itos = {i: token for i, token in enumerate(bpe_tokens)}
bpe_vocab_size = len(bpe_tokens)

# rank dict: lower rank = higher merge priority (earlier in merges.txt)
merge_rank = {pair: i for i, pair in enumerate(merges)}


def _bpe_merge_word(symbols):
    """Apply merges in priority order until no mergeable pair remains.
    O(word_length^2) instead of looping over every merge rule per word."""
    while len(symbols) > 1:
        best_rank, best_i = None, None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            r = merge_rank.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank, best_i = r, i
        if best_i is None:
            break
        symbols = symbols[:best_i] + [symbols[best_i] + symbols[best_i + 1]] + symbols[best_i + 2:]
    return symbols


from functools import lru_cache

@lru_cache(maxsize=None)
def _bpe_merge_word_cached(word, marker):
    return tuple(_bpe_merge_word(list(word) + [marker]))


def encode(text):
    token_ids = []
    for word, marker in split_words_with_sep(text):
        symbols = _bpe_merge_word_cached(word, marker)
        for token in symbols:
            if token in stoi:
                token_ids.append(stoi[token])
    return token_ids


def decode(token_ids):
    tokens = [itos[i] for i in token_ids]
    text = ''.join(tokens)
    text = text.replace('<para>', '\n\n').replace('<end>', ' ')
    return text.strip()


# ── Model architecture (must match training exactly) ─────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        v = self.value(x)
        return weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLM(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )
        self.ln_f    = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.token_embedding_table.weight = self.lm_head.weight  # weight tying

    def forward(self, index, targets=None):
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(index.size(1), device=index.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            index = torch.cat([index, next_token], dim=1)
        return index