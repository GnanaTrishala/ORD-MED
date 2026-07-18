#!/usr/bin/env python
"""
Complete end-to-end pipeline verification script for the ORD-MED repository.
Generates a mock dataset, runs a single-epoch train loop, validation check, and inference.
"""

import os
import sys
import shutil
import subprocess
import torch
import numpy as np
import pandas as pd
from PIL import Image

from config import Config
from models import build_model


def setup_mock_dataset():
    print("Setting up temporary mock dataset...")
    temp_dir = "temp_data"
    img_dir = os.path.join(temp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 1. Create mock images
    num_samples = 4
    image_names = [f"eye_{i}.png" for i in range(num_samples)]
    for name in image_names:
        img_np = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img_np).save(os.path.join(img_dir, name))

    # 2. Create mock CSVs
    data = {
        "image_id": image_names,
        "dr_grade": [0, 2, 4, 1],
        "dme_stage": [0, 1, 2, 1]
    }
    df = pd.DataFrame(data)
    
    # Split mock data into train/val/test
    df.iloc[:2].to_csv(os.path.join(temp_dir, "train.csv"), index=False)
    df.iloc[2:4].to_csv(os.path.join(temp_dir, "val.csv"), index=False)
    df.iloc[2:4].to_csv(os.path.join(temp_dir, "test.csv"), index=False)

    # 3. Create mock config YAML file
    config_yaml = """
encoder:
  name: "efficientnet_b0"
  pretrained: false
  checkpoint_path: null
  freeze_features: false
  dropout: 0.1

heads:
  dr_num_classes: 5
  dme_num_classes: 3
  use_evidential: true
  projection_dim: 512

loss:
  ordinal_method: "corn"
  dme_loss_type: "bce"
  dme_class_weights: [1.0, 1.0, 1.0]
  focal_gamma: 2.0
  evidential_loss_type: "mse"
  lambda1: 1.0
  lambda2: 1.0
  lambda3: 0.5

dataset:
  data_dir: "temp_data/images"
  train_csv: "temp_data/train.csv"
  val_csv: "temp_data/val.csv"
  test_csv: "temp_data/test.csv"
  image_size: 512
  batch_size: 2
  num_workers: 0
  pin_memory: false

trainer:
  epochs: 1
  lr: 0.0001
  weight_decay: 0.00001
  optimizer: "AdamW"
  lr_scheduler: "CosineAnnealingLR"
  device: "cpu"
  seed: 42
  save_dir: "temp_outputs/checkpoints"
  checkpoint_path: null
  log_dir: "temp_outputs/logs"
  project_name: "ORD-MED-TEST"
  experiment_name: "test_run"
  use_amp: false
  patience: 2

referral:
  dr_severity_threshold: 2
  dme_severity_threshold: 1
  uncertainty_threshold: 0.4

visualization:
  use_gradcam: true
  cam_target_layers: ["shared_projection"]
"""
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)

    print("Mock dataset and configuration successfully generated.")
    return temp_dir, config_path


def run_command(cmd, desc):
    print(f"\n>>> Executing command: {' '.join(cmd)} ({desc})...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[FAIL] {desc} failed!")
        print("Stdout:\n", result.stdout)
        print("Stderr:\n", result.stderr)
        raise RuntimeError(f"Command failed: {desc}")
    print(f"[SUCCESS] {desc} completed successfully!")
    return result.stdout


def verify_all():
    temp_dir = None
    try:
        # Step 1: Set up temp directory
        temp_dir, config_path = setup_mock_dataset()

        # Step 2: Initialize model and check simple forward pass programmatically
        print("\n>>> Verifying Model Initialization and Forward Pass...")
        config = Config.load_from_yaml(config_path)
        model = build_model(config)
        model.eval()
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            outputs = model(dummy_input)
        assert "dr_logits" in outputs, "Model forward pass missing 'dr_logits'"
        assert "dme_logits" in outputs, "Model forward pass missing 'dme_logits'"
        print(f"[SUCCESS] Model initialized. Forward shapes: "
              f"DR: {outputs['dr_logits'].shape}, DME: {outputs['dme_logits'].shape}")

        # Step 3: Run Training Script (1 epoch)
        train_cmd = [sys.executable, "train.py", "--config", config_path]
        run_command(train_cmd, "ORD-MED Training Loop")

        # Confirm checkpoint is created
        checkpoint_path = os.path.join("temp_outputs", "checkpoints", "test_run_best.pth")
        if not os.path.exists(checkpoint_path):
            # Fall back to checking latest
            checkpoint_path = os.path.join("temp_outputs", "checkpoints", "test_run_latest.pth")
        assert os.path.exists(checkpoint_path), f"Checkpoint not saved to {checkpoint_path}"
        print(f"Verified training checkpoint output exists: {checkpoint_path}")

        # Step 4: Run Evaluation Script
        eval_cmd = [
            sys.executable, "evaluate.py", 
            "--config", config_path, 
            "--checkpoint", checkpoint_path,
            "--split", "val",
            "--output_dir", "temp_outputs/"
        ]
        run_command(eval_cmd, "ORD-MED Evaluation Pipeline")

        # Confirm evaluation outputs are saved
        assert os.path.exists("temp_outputs/predictions/predictions_val.csv"), "Missing evaluation predictions CSV"
        assert os.path.exists("temp_outputs/metrics/metrics_val.json"), "Missing evaluation metrics JSON"
        assert os.path.exists("temp_outputs/plots/dr_confusion_matrix.png"), "Missing evaluation figures"
        print("Verified evaluation prediction spreadsheets, JSON summaries, and visual figures exist.")

        # Step 5: Run Inference Script
        # Test folder inference
        inf_folder_cmd = [
            sys.executable, "predict.py",
            "--config", config_path,
            "--checkpoint", checkpoint_path,
            "--image_path", "temp_data/images",
            "--output_dir", "temp_outputs/predictions/",
            "--gradcam"
        ]
        run_command(inf_folder_cmd, "Folder Batch Inference with Grad-CAM")

        # Test single image inference
        inf_single_cmd = [
            sys.executable, "predict.py",
            "--config", config_path,
            "--checkpoint", checkpoint_path,
            "--image_path", "temp_data/images/eye_0.png",
            "--output_dir", "temp_outputs/predictions/"
        ]
        run_command(inf_single_cmd, "Single Image Inference")

        print("\n=======================================================")
        print("[SUCCESS] All ORD-MED pipelines verified successfully!")
        print("=======================================================")

    finally:
        # Clean up mock directories
        print("\nCleaning up temporary files...")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists("temp_outputs"):
            shutil.rmtree("temp_outputs")
        print("Cleanup completed.")


if __name__ == "__main__":
    verify_all()
