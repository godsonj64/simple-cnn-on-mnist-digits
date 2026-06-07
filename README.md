# Simple CNN on MNIST Digits

This project trains an image-recognition model that reads handwritten digits (0-9) using the classic MNIST-style dataset arranged as image folders.

- **Task:** image classification (10 classes)
- **Recommended model:** EfficientNet-B0 (transfer learning)
- **Baseline model:** small CNN trained from scratch
- **Metrics:** accuracy, F1
- **Export formats:** ONNX, TorchScript
- **Epochs:** 20

## Dataset format

The dataset uses the `image_folder` layout. See [data/README.md](data/README.md) for details. In short:

```
data/
  train/
    0/ img1.png ...
    1/ ...
    ...
    9/ ...
  val/
    0/ ...
    ...
    9/ ...
```

## Quickstart

```bash
pip install -r requirements.txt

# Train
bash scripts/run_train.sh

# Evaluate
python -m src.evaluate --config configs/default.yaml --checkpoint outputs/best.pt

# Export to ONNX + TorchScript
python -m src.export --config configs/default.yaml --checkpoint outputs/best.pt
```

## Choosing the model

Set `model.name` in `configs/default.yaml` to either:

- `efficientnet_b0` — recommended transfer-learning model (pretrained backbone).
- `small_cnn` — lightweight baseline trained from scratch.

## Training output

Each epoch prints one parseable line:

```
epoch 1/20 loss=0.3124 val_acc=0.9521
```

## Docker

```bash
docker build -t mnist-cnn .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs mnist-cnn
```
