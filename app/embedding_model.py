import math
import spacy

class EmbeddingModel:
    """Looks up static word vectors from a spaCy model."""

    def __init__(self, model_name: str = "en_core_web_md"):
        # Load once; parser and NER aren't needed for vector lookup.
        self.nlp = spacy.load(model_name, disable=["parser", "ner"])

    def get_embedding(self, word: str) -> list[float] | None:
        doc = self.nlp(word.strip())
        if len(doc) != 1:
            raise ValueError("Send exactly one word, for example 'king'.")
        token = doc[0]
        if not token.has_vector:
            return None
        return token.vector.tolist()

    def similarity(self, word1: str, word2: str) -> float | None:
        a = self.get_embedding(word1)
        b = self.get_embedding(word2)
        if a is None or b is None:
            return None
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b)