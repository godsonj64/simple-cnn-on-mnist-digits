import argparse

import torch
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import load_config, resolve_device


def evaluate(config_path, checkpoint_path):
    cfg = load_config(config_path)
    device = resolve_device(cfg["train"]["device"])

    _, val_loader, _ = build_dataloaders(cfg["data"], cfg["train"]["batch_size"])

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")

    print(f"accuracy={acc:.4f}")
    print(f"f1={f1:.4f}")
    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate the digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
