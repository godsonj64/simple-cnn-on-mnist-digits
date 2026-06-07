import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import ensure_dir, load_config, resolve_device, set_seed


def evaluate_accuracy(model, loader, device):
    """Compute validation accuracy."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    if not targets:
        return 0.0
    return accuracy_score(targets, preds)


def train(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])

    train_loader, val_loader, _ = build_dataloaders(
        cfg["data"], cfg["train"]["batch_size"]
    )

    model = build_model(cfg["model"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    out_dir = cfg["output"]["dir"]
    ensure_dir(out_dir)
    best_path = os.path.join(out_dir, cfg["output"]["best_checkpoint"])

    total_epochs = cfg["train"]["epochs"]
    best_acc = -1.0

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        val_acc = evaluate_accuracy(model, val_loader, device)

        print(
            f"epoch {epoch}/{total_epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "config": cfg},
                best_path,
            )

    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train the digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
