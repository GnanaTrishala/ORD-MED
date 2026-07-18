#!/usr/bin/env python
"""
Verification script for the integrated data pipeline of the ORD-MED project.
Loads the integrated data, initializes a DataLoader, pulls a batch, and verifies shapes.
"""

import sys
import torch

from config import Config
from datasets.dataset import get_dataloaders


def verify_integrated_dataloader():
    print("Initializing baseline configuration...")
    config = Config()
    
    # 1. Update config parameters to use the newly generated integrated datasets
    config.dataset.train_csv = "dataset/integrated_train.csv"
    config.dataset.val_csv = "dataset/integrated_val.csv"
    config.dataset.test_csv = "dataset/integrated_test.csv"
    config.dataset.batch_size = 4
    config.dataset.num_workers = 0  # CPU-friendly for verification
    config.dataset.pin_memory = False

    print("Building PyTorch dataloaders for the integrated datasets...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Dataloaders initialized:")
    print(f"  Train split size: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val split size:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")

    # 2. Extract and inspect a single batch
    print("\nExtracting one batch from train_loader...")
    batch_idx, batch = next(enumerate(train_loader))
    
    images = batch["image"]
    dr_labels = batch["dr_label"]
    dme_labels = batch["dme_label"]
    paths = batch["image_path"]

    # 3. Verify tensor shapes and data types
    print("\nVerifying batch tensor dimensions:")
    print(f"  Image batch shape: {images.shape} (Expected: [4, 3, 512, 512])")
    print(f"  DR label shape:    {dr_labels.shape} (Expected: [4])")
    print(f"  DME label shape:   {dme_labels.shape} (Expected: [4])")
    print(f"  DR labels values:  {dr_labels.tolist()} (Type: {dr_labels.dtype})")
    print(f"  DME labels values: {dme_labels.tolist()} (Type: {dme_labels.dtype})")

    # Assertions
    assert images.shape == torch.Size([4, 3, 512, 512]), "Incorrect image batch shape!"
    assert dr_labels.shape == torch.Size([4]), "Incorrect DR label shape!"
    assert dme_labels.shape == torch.Size([4]), "Incorrect DME label shape!"
    
    # Verify that DME contains -100 placeholders for APTOS samples
    dme_list = dme_labels.tolist()
    has_masked = -100 in dme_list
    print(f"  Contains masked targets (-100)? {has_masked}")

    print("\n=======================================================")
    print("[SUCCESS] Data pipeline verification completed successfully!")
    print("=======================================================")


if __name__ == "__main__":
    try:
        verify_integrated_dataloader()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] Data pipeline verification failed! Error: {str(e)}")
        sys.exit(1)
