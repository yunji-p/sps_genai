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

## Handwritten digit generation (Assignment 3)

`POST /generate-image` takes a count and an optional seed and returns a PNG
grid of GAN-generated MNIST digits.

The model is the GAN in `helper_lib/model.py` (Generator: 100-d noise, Linear
to 7x7x128, two ConvTranspose2d blocks to 28x28, Tanh; Discriminator: two
Conv2d blocks with LeakyReLU(0.2), Linear to one score), returned by
`get_model("GAN")`. Training is `train_gan` in `helper_lib/trainer.py`,
sampling is `generate_samples` in `helper_lib/generator.py`.

Train with `python -m training.train_gan`. Trained generator weights are in
`app/weights/gan_mnist_generator.pth`.

    curl -X POST http://127.0.0.1:8000/generate-image \
      -H "Content-Type: application/json" \
      -d '{"num_images": 16, "seed": 42}' --output digits.png