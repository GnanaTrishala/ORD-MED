#!/usr/bin/env python
"""
Gradio Web Application for the ORD-MED research demonstration.
Provides an interactive portal to upload retinal images and visualize multi-task predictions,
subjective uncertainty, automated referrals, and explainable Grad-CAM heatmaps.
"""

import os
import argparse
import numpy as np
import torch
import cv2
import gradio as gr
from PIL import Image

from config import Config
from models import build_model
from datasets.transforms import get_inference_transforms
from utils.checkpoint import load_checkpoint
from inference import Predictor
from visualization.gradcam import ModelTaskWrapper

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None
    show_cam_on_image = None


def generate_cam_overlay(model, image_np, transforms, task_name, target_layers, device):
    """Generates a Grad-CAM heatmap overlay array for a given task."""
    if GradCAMPlusPlus is None:
        return None

    # Preprocess image for model input
    augmented = transforms(image=image_np)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Wrap model to isolate specific task output
    wrapped_model = ModelTaskWrapper(model, task=task_name)
    wrapped_model.eval()

    # Resolve target layers
    resolved_layers = []
    for layer_name in target_layers:
        try:
            layer = model
            for part in layer_name.split("."):
                layer = getattr(layer, part)
            resolved_layers.append(layer)
        except AttributeError:
            pass
            
    if not resolved_layers:
        # Fallback to first Conv2d layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                resolved_layers = [module]
        resolved_layers = [resolved_layers[-1]] if resolved_layers else []

    if not resolved_layers:
        return None

    # Get target class for optimization
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs["dr_logits"] if task_name == "dr" else outputs["dme_logits"]
        target_class = torch.argmax(logits, dim=-1).item()

    # Instantiate Grad-CAM engine
    cam = GradCAMPlusPlus(model=wrapped_model, target_layers=resolved_layers)
    targets = [ClassifierOutputTarget(target_class)]

    # Generate CAM map
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # Resize raw image to 512x512 to match heatmap overlay dimensions
    resized_orig = cv2.resize(image_np, (512, 512)) / 255.0

    # Overlay
    cam_image = show_cam_on_image(resized_orig, grayscale_cam, use_rgb=True)
    return cam_image, target_class


def build_demo(config_path, checkpoint_path):
    # Load configuration
    config = Config.load_from_yaml(config_path)
    
    # Set device
    device = torch.device(config.trainer.device if torch.cuda.is_available() else "cpu")

    # Build model and restore checkpoints
    print("Loading model for Gradio app...")
    model = build_model(config)
    model, _, _ = load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    # Setup Predictor
    transforms = get_inference_transforms(config)
    predictor = Predictor(model=model, transforms=transforms, device=device, config=config)

    # Define prediction wrapper for Gradio
    def process_image(input_img):
        if input_img is None:
            return None, None, None, "Please upload a retinal image."

        # Convert to numpy array
        img_np = np.array(input_img)

        # 1. Run predictions
        results = predictor.predict(img_np)

        # 2. Format outcomes text
        output_html = f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f7f9fa; border: 1px solid #e1e8ed;">
            <h3 style="margin-top: 0; color: #1da1f2;">Diagnostic Evaluation</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #e1e8ed;">DR Severity Grade:</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e1e8ed;">Grade {results['DR Grade']} ({results['DR Confidence']*100:.1f}% Confidence)</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #e1e8ed;">DME Stage:</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e1e8ed;">Stage {results['DME']} ({results['DME Confidence']*100:.1f}% Confidence)</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold; border-bottom: 1px solid #e1e8ed;">Epistemic Uncertainty:</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e1e8ed;">{results['Uncertainty']:.4f}</td>
                </tr>
                <tr style="background-color: {'#ffebee' if results['Referral Decision'] else '#e8f5e9'};">
                    <td style="padding: 8px; font-weight: bold;">Referral Recommendation:</td>
                    <td style="padding: 8px; font-weight: bold; color: {'#c62828' if results['Referral Decision'] else '#2e7d32'};">
                        {'⚠️ REFERRAL RECOMMENDED' if results['Referral Decision'] else '✅ SAFE (NO REFERRAL REQUIRED)'}
                    </td>
                </tr>
            </table>
        </div>
        """

        # 3. Generate Grad-CAM heatmaps
        dr_cam, dr_class = generate_cam_overlay(
            model=model,
            image_np=img_np,
            transforms=transforms,
            task_name="dr",
            target_layers=config.visualization.cam_target_layers,
            device=device
        )
        
        dme_cam, dme_class = generate_cam_overlay(
            model=model,
            image_np=img_np,
            transforms=transforms,
            task_name="dme",
            target_layers=config.visualization.cam_target_layers,
            device=device
        )

        return dr_cam, dme_cam, output_html

    # Build the block layout interface
    with gr.Blocks(title="ORD-MED Research Demonstration") as demo:
        gr.Markdown(
            """
            # ORD-MED (Ordinal Multi-task Evidential Diabetic Eye Disease Network)
            ### Clinical Research & Interpretability Demonstration Portal
            
            This portal demonstrates the joint evaluation of **Diabetic Retinopathy (DR)** severity grading and **Diabetic Macular Edema (DME)** severity staging from color fundus images. 
            The model incorporates **evidential deep learning** to estimate prediction uncertainty and triggers automated referrals for highly ambiguous cases.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Retinal Fundus Image")
                btn = gr.Button("Execute Diagnostics", variant="primary")
            
            with gr.Column(scale=2):
                output_report = gr.HTML(label="Diagnostic Report Summary")
                
        with gr.Row():
            with gr.Column():
                dr_gradcam = gr.Image(label="DR Grad-CAM++ Explainability Map")
            with gr.Column():
                dme_gradcam = gr.Image(label="DME Grad-CAM++ Explainability Map")
                
        btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[dr_gradcam, dme_gradcam, output_report]
        )

        gr.Markdown(
            """
            *Note: This portal is for research and demonstration purposes only. It is not approved for actual clinical diagnostics.*
            """
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORD-MED Gradio App Launch Utility")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true", help="Launch app with sharing link enabled")
    args = parser.parse_args()

    demo = build_demo(args.config, args.checkpoint)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)
