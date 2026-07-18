import os
from typing import List, Any
import numpy as np
import cv2
import torch
import torch.nn as nn

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    # Fallback placeholders in case package is not yet installed
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None
    show_cam_on_image = None


class ModelTaskWrapper(nn.Module):
    """
    Utility wrapper to isolate a single task output from the multi-task model
    so that standard Grad-CAM libraries (which expect a single tensor output) 
    can target individual task heads.
    """
    def __init__(self, model: nn.Module, task: str = "dr"):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        if self.task == "dr":
            return outputs["dr_logits"]
        else:
            return outputs["dme_logits"]


def run_gradcam(
    model: nn.Module,
    image_path: str,
    transforms: Any,
    target_layers: List[str],
    save_dir: str,
    device: torch.device
) -> None:
    """
    Generates and saves Grad-CAM heatmaps for both DR and DME diagnosis channels.

    Args:
        model (nn.Module): The trained multi-task model.
        image_path (str): Filepath of the original input image.
        transforms (callable): Preprocessing transforms.
        target_layers (list of str): Layer names in the backbone to extract gradients from.
        save_dir (str): Directory where output heatmaps will be written.
        device (torch.device): Execution device.
    """
    if GradCAMPlusPlus is None:
        print("Warning: 'pytorch-grad-cam' is not installed. Skipping Grad-CAM generation.")
        return

    # 1. Load image and preprocess
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize raw image to match model input for overlay
    h, w, _ = orig_img.shape
    resized_orig = cv2.resize(orig_img, (512, 512)) / 255.0

    # Get input tensor
    augmented = transforms(image=orig_img)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # 2. Identify the target layers in the model
    # Usually we target the last convolutional layer of the backbone
    # Let's assume target_layers contains references we need to resolve
    # e.g. model.backbone.conv_head
    resolved_layers = []
    for layer_name in target_layers:
        try:
            # Simple attribute resolution
            layer = model
            for part in layer_name.split("."):
                layer = getattr(layer, part)
            resolved_layers.append(layer)
        except AttributeError:
            # Fallback to last block of backbone
            pass
            
    if not resolved_layers:
        # Fallback search for a convolutional layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                resolved_layers = [module]
        # Get the last one
        resolved_layers = [resolved_layers[-1]] if resolved_layers else []

    if not resolved_layers:
        print("Error: Could not locate a convolutional layer for Grad-CAM.")
        return

    # 3. Generate visualizer heatmaps for both tasks
    for task_name in ["dr", "dme"]:
        # Wrap model to isolate the specific task head output
        wrapped_model = ModelTaskWrapper(model, task=task_name)
        wrapped_model.eval()

        # Instantiate Grad-CAM engine
        cam = GradCAMPlusPlus(model=wrapped_model, target_layers=resolved_layers)

        # Retrieve prediction category to align visualization target
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs["dr_logits"] if task_name == "dr" else outputs["dme_logits"]
            target_class = torch.argmax(logits, dim=-1).item()

        targets = [ClassifierOutputTarget(target_class)]

        # Generate grayscale CAM map
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Overlay heatmap on original image
        cam_image = show_cam_on_image(resized_orig, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # Save resulting file
        img_name = os.path.basename(image_path).rsplit(".", 1)[0]
        output_file = os.path.join(save_dir, f"{img_name}_gradcam_{task_name}_class_{target_class}.png")
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(output_file, cam_image)
        
        print(f"Saved {task_name.upper()} Grad-CAM to: {output_file}")
