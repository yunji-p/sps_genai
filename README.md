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