# MNIST Digit Classifier

A CNN and a plain MLP trained from scratch with PyTorch. Both use the same data split, batches and 12 epochs. No pretrained weights are used.

**Results:** CNN 99.02%, MLP 97.92% on 10,000 test images. The saved CNN also predicts seven new handwritten images.

See [START.md](START.md) to run the project and [REPORT.md](REPORT.md) for results and image preprocessing.

## Data and training

MNIST is loaded with `torchvision.datasets.MNIST`. Pixels are converted to float32 and divided by 255. The split is 54,001 training, 5,999 validation and 10,000 test images. Validation is a class-balanced proportion of the original training set; the test set stays separate.

Both models use seed 42, Adam, learning rate 0.001, batch size 128 and cross-entropy loss. Batch order is the same for both models. Each runs for 12 epochs. The checkpoint with the lowest validation loss is then tested.

## CNN layers

| Layer | Settings | Output |
|---|---|---|
| Input | Grayscale | 1 × 28 × 28 |
| Conv2d | 1 → 16, kernel 3, padding 1 | 16 × 28 × 28 |
| ReLU | Activation | 16 × 28 × 28 |
| MaxPool2d | Kernel 2, stride 2 | 16 × 14 × 14 |
| Conv2d | 16 → 32, kernel 3, padding 1 | 32 × 14 × 14 |
| ReLU | Activation | 32 × 14 × 14 |
| MaxPool2d | Kernel 2, stride 2 | 32 × 7 × 7 |
| Flatten | — | 1568 |
| Linear | 1568 → 128 | 128 |
| ReLU | Activation | 128 |
| Linear | 128 → 10 | 10 logits |

MLP: `Flatten → Linear(784,256) → ReLU → Linear(256,64) → ReLU → Linear(64,10)`.

The CNN has 206,922 parameters; the MLP has 218,058. Equal training settings do not mean equal computing cost.

## Files

- `models.py`: network layers.
- `train.py`: training, test evaluation and plots.
- `predict.py`: load saved weights and predict one image.
- `external_test.py`: test every labeled image in `custom_images/manifest.json`.
- `app.py`, `static/index.html`: optional upload page and Flask endpoint.
- `show_filters.py`: plot the first convolution layer.
- `artifacts/`: saved weights, results and plots.

The seven custom images are user-provided drawings, not MNIST samples. Their labels are 3, 8, 1, 4, 5, 7, 7. Add new filenames and labels to the manifest before testing them. Inference uses `eval()` and no gradients; it does not retrain the model.

Source: [MNIST](http://yann.lecun.com/exdb/mnist/). Prepared with AI assistance.
