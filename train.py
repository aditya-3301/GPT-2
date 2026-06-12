import torch
import re
from collections import defaultdict

text = open("train.txt", "r", encoding="utf-8").read()

unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)


print(''.join(unique_chars))
print(f"\nvocab_Size:{vocab_size}")

# unique_chars.extend(['4','6','7','9','0'])
# Adding remaining numbers 

def word_tokenize(text):
    """Each word is a token, but also each special character, spaces arent ocnsidered."""
    # Split on spaces but keep punctuation as separate tokens
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens
word_tokens = word_tokenize(text)


print(f"\nFirst 50 WORD tokens:\n {word_tokens[:50]}")

def get_vocab(text):
    """Convert words into character seq. with END marker for Bytepairencoding"""
    vocab = defaultdict(int)
    for word in text.strip().split():
        # 'test' becomes ('t','e','s','t','<end>')
        vocab[' '.join(list(word)) + ' <end>'] += 1
    return vocab

def get_stats(vocab):
    """Count frequency of every adjacent pair and stores in dict pairs"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_mostfrequent(best_pair, vocab):
    """Merge the most frequent pair across all words, variable- pair is the most freq adjacent pair"""
    new_vocab = {}
    to_replace = ' '.join(best_pair)
    replacement = ''.join(best_pair)
    for word in vocab:
        new_word = word.replace(to_replace, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab


def train_bpe(text, n_merges):
    vocab = get_vocab(text)
    merges = []

    for i in range(n_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        # merge the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_mostfrequent(best_pair, vocab)
        merges.append(best_pair)

    return vocab, merges

print("\n\n")
bpe_vocab, merges = train_bpe(text, n_merges=100)

# Build final token list from BPE vocab
bpe_tokens = set()
for word in bpe_vocab:
    for token in word.split():
        bpe_tokens.add(token)

bpe_tokens = sorted(list(bpe_tokens))
print(f"BPE vocab size: {len(bpe_tokens)}")
print(f"First 20 tokens: {bpe_tokens[:20]}")

stoi = {token: i for i, token in enumerate(bpe_tokens)}
itos = {i: token for i, token in enumerate(bpe_tokens)}

#string ot integer and integer to string

def encode(text):
    words = text.strip().split()
    token_ids = []
    for word in words:
        # split word into characters with end marker
        symbols = list(word) + ['<end>']
        # apply each merge in order
        for pair in merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                    symbols = symbols[:i] + [''.join(pair)] + symbols[i+2:]
                else:
                    i += 1
        # look up each resulting token in stoi
        for token in symbols:
            if token in stoi:
                token_ids.append(stoi[token])
    return token_ids

encoded = encode(text)

n = int(0.9 * len(encoded))
train = encoded[:n]
test = encoded[n:]


train = torch.tensor(train, dtype=torch.long)
test = torch.tensor(test, dtype=torch.long)

print(f"Train tokens: {len(train)}, Test tokens: {len(test)}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")