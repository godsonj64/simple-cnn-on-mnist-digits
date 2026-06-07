import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Load a YAML config file into a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Make runs reproducible by seeding all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str):
    """Resolve the desired device string into an actual torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
