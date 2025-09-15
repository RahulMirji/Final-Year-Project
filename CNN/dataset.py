from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_dir: str = "dataset"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    augment: bool = True


def build_transforms(img_size: int, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_tfms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ] if augment else [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transforms.Compose(train_tfms), eval_tfms, eval_tfms


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")
    test_dir = os.path.join(cfg.data_dir, "test")

    train_tfms, val_tfms, test_tfms = build_transforms(cfg.img_size, cfg.augment)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=test_tfms)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes
