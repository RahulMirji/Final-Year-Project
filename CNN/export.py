from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from model import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    num_classes = ckpt.get('num_classes')
    img_size = ckpt.get('img_size', 224)

    model = build_model(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(DEVICE)
    return model, img_size


def export_onnx(model: torch.nn.Module, img_size: int, out_path: str):
    dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=['input'], output_names=['logits'],
        opset_version=12, do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    print(f"Exported ONNX model to {out_path}")


def export_torchscript(model: torch.nn.Module, img_size: int, out_path: str):
    dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, out_path)
    print(f"Exported TorchScript model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model to ONNX or TorchScript")
    parser.add_argument('--ckpt', type=str, default=os.path.join('models', 'checkpoint.pth'))
    parser.add_argument('--onnx', type=str, default=os.path.join('models', 'model.onnx'))
    parser.add_argument('--ts', type=str, default=os.path.join('models', 'model.ts'))
    parser.add_argument('--format', type=str, choices=['onnx', 'ts', 'both'], default='both')

    args = parser.parse_args()

    model, img_size = load_checkpoint(args.ckpt)
    Path('models').mkdir(parents=True, exist_ok=True)

    if args.format in ('onnx', 'both'):
        export_onnx(model, img_size, args.onnx)
    if args.format in ('ts', 'both'):
        export_torchscript(model, img_size, args.ts)
