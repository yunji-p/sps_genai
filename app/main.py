from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(title="SPS GenAI API")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel("en_core_web_md")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    try:
        vector = embedding_model.get_embedding(request.word)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if vector is None:
        raise HTTPException(status_code=404, detail=f"'{request.word}' has no vector in the model")
    return {"word": request.word, "dimension": len(vector), "embedding": vector}


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    try:
        score = embedding_model.similarity(request.word1, request.word2)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if score is None:
        raise HTTPException(status_code=404, detail="One of the words has no vector in the model")
    return {"word1": request.word1, "word2": request.word2, "cosine_similarity": score}