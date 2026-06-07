import argparse

import torch

from src.model import build_model
from src.utils import ensure_dir, load_config, resolve_device


def export(config_path, checkpoint_path):
    cfg = load_config(config_path)
    device = resolve_device("cpu")

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ensure_dir(cfg["output"]["dir"])
    image_size = cfg["data"]["image_size"]
    dummy = torch.randn(1, 3, image_size, image_size)

    onnx_path = cfg["export"]["onnx_path"]
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=cfg["export"]["opset"],
    )
    print(f"Saved ONNX model to {onnx_path}")

    ts_path = cfg["export"]["torchscript_path"]
    scripted = torch.jit.trace(model, dummy)
    scripted.save(ts_path)
    print(f"Saved TorchScript model to {ts_path}")

    return onnx_path, ts_path


def main():
    parser = argparse.ArgumentParser(description="Export the digit classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best.pt")
    args = parser.parse_args()
    export(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
