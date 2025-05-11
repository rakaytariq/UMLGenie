import json
from collections import Counter
from typing import List

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]

class Tokenizer:
    def __init__(self, vocab=None):
        self.word2idx = {}
        self.idx2word = {}

        if vocab is not None and len(vocab) > 0:
            self.vocab = vocab
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        else:
            self.vocab = []

    def build_vocab(self, corpus: List[str], max_size=10000, min_freq=1):
        counter = Counter()
        for text in corpus:
            tokens = text.strip().split()
            counter.update(tokens)

        most_common = [w for w, f in counter.items() if f >= min_freq][:max_size]

        # Filter out bad tokens like </eos> or <unk> added from corrupted output
        cleaned = [
            w for w in most_common
            if w not in SPECIAL_TOKENS and not w.startswith("</") and not w.startswith("<")
        ]

        self.vocab = SPECIAL_TOKENS + cleaned
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        tokens = text.strip().split()
        ids = [self.word2idx.get(token, self.word2idx[UNK]) for token in tokens]
        if add_special_tokens:
            ids = [self.word2idx[SOS]] + ids + [self.word2idx[EOS]]
        return ids

    def decode(self, ids: List[int], skip_special=True) -> str:
        tokens = [self.idx2word.get(i, UNK) for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return ' '.join(tokens)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
