from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import DataConfig, build_dataloaders
from model import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model: nn.Module, loader, criterion, optimizer) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs.detach(), labels) * images.size(0)

    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def evaluate(model: nn.Module, loader, criterion) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def main(args: argparse.Namespace) -> None:
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        augment=not args.no_augment,
    )

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(data_cfg)

    model = build_model(num_classes=num_classes, in_channels=3, dropout=args.dropout).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    ckpt_path = Path(args.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'img_size': args.img_size,
            }, ckpt_path)
            print(f"Saved new best checkpoint to {ckpt_path} (acc={best_val_acc:.4f})")

    # Final test
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple CNN on an ImageFolder dataset")
    parser.add_argument('--data-dir', type=str, default='dataset', help='Dataset root with train/val/test folders')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--ckpt-path', type=str, default=os.path.join('models', 'checkpoint.pth'))
    parser.add_argument('--no-augment', action='store_true', help='Disable training augmentation')

    args = parser.parse_args()
    main(args)
