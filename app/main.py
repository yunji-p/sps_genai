from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .image_generator import DigitGenerator

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

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from app.image_classifier import ImageClassifier

image_classifier = ImageClassifier()  # load the trained CNN once, at startup

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        result = image_classifier.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a readable image.")
    return {"filename": file.filename, **result}

_digit_generator: DigitGenerator | None = None


def get_digit_generator() -> DigitGenerator:
    """Load on first request so the app still boots if the weights are missing."""
    global _digit_generator
    if _digit_generator is None:
        try:
            _digit_generator = DigitGenerator()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _digit_generator


class DigitRequest(BaseModel):
    num_images: int = Field(16, ge=1, le=64, description="How many digits to generate")
    seed: int | None = Field(None, description="Fix the seed for a reproducible image")


@app.post("/generate-image")
def generate_image(request: DigitRequest):
    """Generate handwritten digits with the trained GAN, returned as one PNG grid."""
    png = get_digit_generator().generate_png(request.num_images, request.seed)
    return StreamingResponse(png, media_type="image/png")