#!/usr/bin/env python
"""
Inference script for running predictions on single or folders of ophthalmic fundus images.
Generates diagnostic classifications, evidential uncertainties, referral choices, and Grad-CAMs.
"""

import os
import argparse
import pandas as pd
import torch

from config import Config
from datasets.transforms import get_inference_transforms
from models import build_model
from utils.checkpoint import load_checkpoint
from inference import Predictor
from visualization.gradcam import run_gradcam


def main():
    parser = argparse.ArgumentParser(description="ORD-MED Image Inference Pipeline")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input retinal image or folder")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions/", help="Directory to save predictions")
    parser.add_argument("--gradcam", action="store_true", help="Generate and save Grad-CAM heatmaps")
    args = parser.parse_args()

    # 1. Load configuration
    config = Config.load_from_yaml(args.config)
    
    if args.output_dir != "outputs/predictions/":
        config.outputs.predictions = args.output_dir
        # Resolve gradcam to be adjacent to customized predictions dir
        gradcam_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "gradcam")
    else:
        args.output_dir = config.outputs.predictions
        gradcam_dir = config.outputs.gradcam

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)
    figures_dir = gradcam_dir

    # 2. Build model and load weights
    print("Building model architecture...")
    model = build_model(config)
    print(f"Loading checkpoint weights from: {args.checkpoint}...")
    model, _, _ = load_checkpoint(model, args.checkpoint)
    
    device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 3. Setup Predictor API
    transforms = get_inference_transforms(config)
    engine = Predictor(model=model, transforms=transforms, device=device, config=config)

    # 4. Handle Single Image vs. Folder Prediction
    is_dir = os.path.isdir(args.image_path)
    
    if is_dir:
        print(f"Scanning folder for predictions: {args.image_path}...")
        csv_output_path = os.path.join(args.output_dir, "batch_predictions.csv")
        df_results = engine.predict_folder(args.image_path, output_csv_path=csv_output_path)
        
        # Display summary in console
        print(f"\nBatch predictions completed! Processed {len(df_results)} images.")
        print(f"Tabular report saved to: {csv_output_path}")
        
        # Get list of images for Grad-CAM if requested
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        image_files = sorted([
            os.path.join(args.image_path, f) for f in os.listdir(args.image_path)
            if f.lower().endswith(image_extensions)
        ])
    else:
        # Single image prediction
        image_files = [args.image_path]
        try:
            results = engine.predict(args.image_path)
            
            # Print findings
            print(f"\n=== Prediction Results for: {os.path.basename(args.image_path)} ===")
            print(f"  DR Grade (Predicted): {results['DR Grade']}")
            print(f"  DR Confidence:        {results['DR Confidence']:.4f}")
            print(f"  DME Stage (Predicted): {results['DME']}")
            print(f"  DME Confidence:       {results['DME Confidence']:.4f}")
            print(f"  Average Confidence:   {results['Confidence']:.4f}")
            
            if config.heads.use_evidential:
                print(f"  DR Evidential Uncertainty: {results['DR Uncertainty']:.4f}")
                print(f"  DME Evidential Uncertainty: {results['DME Uncertainty']:.4f}")
                print(f"  Average Uncertainty:        {results['Uncertainty']:.4f}")
                print(f"  Referral Decision:          {results['Referral Decision']}")
                
            # Export single prediction result as JSON
            out_filename = os.path.basename(args.image_path).rsplit(".", 1)[0] + "_pred.json"
            engine.save_results(results, os.path.join(args.output_dir, out_filename))
            print(f"Prediction details exported to: {os.path.join(args.output_dir, out_filename)}")

            # Also save as single row CSV for uniformity
            df_single = pd.DataFrame([results])
            csv_output_path = os.path.join(args.output_dir, os.path.basename(args.image_path).rsplit(".", 1)[0] + "_pred.csv")
            df_single.to_csv(csv_output_path, index=False)
            print(f"Tabular report exported to: {csv_output_path}")

        except Exception as e:
            print(f"Error processing single image prediction: {str(e)}")
            image_files = []

    # 5. Generate Grad-CAM heatmaps if requested
    if args.gradcam and len(image_files) > 0:
        print(f"\nGenerating Grad-CAM overlays for {len(image_files)} image(s)...")
        for img_path in image_files:
            try:
                run_gradcam(
                    model=model,
                    image_path=img_path,
                    transforms=transforms,
                    target_layers=config.visualization.cam_target_layers,
                    save_dir=figures_dir,
                    device=device
                )
            except Exception as e:
                print(f"Failed to generate Grad-CAM for {img_path}. Error: {str(e)}")
        print(f"Visual overlays saved inside: {figures_dir}")


if __name__ == "__main__":
    main()
