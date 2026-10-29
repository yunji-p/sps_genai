## Run with Docker
```bash
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```
Docs: http://127.0.0.1:8000/docs

## Endpoints
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | /generate | {"start_word": "bigram", "length": 5} | bigram-generated text |
| POST | /embedding | {"word": "king"} | 300-dim spaCy (en_core_web_md) vector |

`/embedding` accepts one word: multi-word input returns 400, words without a vector return 404.

## Image classification (Assignment 2)

`POST /classify` takes an image upload and returns its CIFAR-10 class
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

The model is the CNN in `app/cnn_model.py` (64x64x3 input, two conv/pool
blocks, FC 100, FC 10), trained with `python -m training.train_cnn`.
Trained weights are in `app/weights/cnn_cifar10.pth`.

    curl -X POST http://127.0.0.1:8000/classify -F "file=@cat.jpg"