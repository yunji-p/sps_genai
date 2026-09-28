import random
import re
from collections import defaultdict, Counter


class BigramModel:
    def __init__(self, corpus):
        self.bigram_counts = defaultdict(Counter)
        for text in corpus:
            tokens = self._tokenize(text)
            for current_word, next_word in zip(tokens, tokens[1:]):
                self.bigram_counts[current_word][next_word] += 1

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        words = [current_word]
        for _ in range(length - 1):
            next_options = self.bigram_counts.get(current_word)
            if not next_options:
                break
            candidates = list(next_options.keys())
            weights = list(next_options.values())
            current_word = random.choices(candidates, weights=weights)[0]
            words.append(current_word)
        return " ".join(words)
