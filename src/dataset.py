import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size, grayscale_to_rgb):
    """Build the image preprocessing pipeline."""
    ops = [transforms.Resize((image_size, image_size))]
    if grayscale_to_rgb:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.append(transforms.ToTensor())
    ops.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(ops)


def build_dataset(data_cfg, split):
    """Build an ImageFolder dataset for the given split ('train' or 'val')."""
    subdir = data_cfg["train_dir"] if split == "train" else data_cfg["val_dir"]
    path = os.path.join(data_cfg["root"], subdir)
    tfm = build_transforms(data_cfg["image_size"], data_cfg["grayscale_to_rgb"])
    return datasets.ImageFolder(path, transform=tfm)


def build_dataloaders(data_cfg, batch_size):
    """Create training and validation dataloaders."""
    train_ds = build_dataset(data_cfg, "train")
    val_ds = build_dataset(data_cfg, "val")
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes
