import os
from typing import Dict, Any, Union, List, Optional
import torch
import numpy as np
import pandas as pd
from PIL import Image

from config import Config
from models.modules.referral_module import ReferralModule


class Predictor:
    """
    Inference Predictor API for the ORD-MED network.
    Provides single-image evaluation, batch/folder predictions, 
    and exports tabular CSV outputs.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        transforms: Any,
        device: torch.device,
        config: Config
    ) -> None:
        """
        Args:
            model (nn.Module): Pre-trained model.
            transforms (callable): Image preprocessing transformations.
            device (torch.device): Device to run inference on.
            config (Config): Centralized configuration instance.
        """
        self.model = model
        self.transforms = transforms
        self.device = device
        self.config = config
        self.model.eval()

        # Initialize referral decision logic module
        self.referral_module = ReferralModule(config=config)

    @torch.no_grad()
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Runs prediction on a single input image.
        
        Args:
            image (str, PIL.Image, or np.ndarray): Input image or file path.

        Returns:
            dict: Containing prediction findings:
                - 'DR Grade': Predicted DR severity level (0-4).
                - 'DME': Predicted DME stage level (0-2).
                - 'Confidence': Overall confidence score (average class probability).
                - 'Uncertainty': Overall epistemic uncertainty (average subjective uncertainty).
                - 'Referral Decision': Boolean flag recommending specialist referral.
                - Task-specific confidences, uncertainties, and raw Dirichlet evidence list.
        """
        # 1. Standardize image input loading
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found at path: {image}")
            img_pil = Image.open(image).convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_np = image
            if img_np.ndim == 2:  # Grayscale conversion
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[0] == 3 and img_np.shape[2] != 3:  # Transpose CHW to HWC
                img_np = np.transpose(img_np, (1, 2, 0))
        else:
            raise TypeError("Unsupported image input type. Provide file path, PIL.Image, or NumPy array.")

        # 2. Apply preprocessing and normalize
        augmented = self.transforms(image=img_np)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)  # Shape: (1, 3, H, W)

        # 3. Model forward pass
        outputs = self.model(input_tensor)

        # 4. Parse softmax probabilities and class predictions
        dr_logits = outputs["dr_logits"]
        dme_logits = outputs["dme_logits"]

        # DME uses standard softmax
        dme_probs = torch.softmax(dme_logits, dim=-1).squeeze(0)
        dme_class = torch.argmax(dme_probs).item()

        # DR uses CORN or EMD
        ordinal_method = getattr(self.config.loss, "ordinal_method", "corn")
        if ordinal_method == "corn":
            num_classes_dr = self.config.heads.dr_num_classes
            corn_logits = dr_logits[:, :num_classes_dr - 1]
            
            sigmoids = torch.sigmoid(corn_logits)
            cum_probs = torch.ones(sigmoids.size(0), num_classes_dr, device=sigmoids.device)
            cum_probs[:, 1:] = torch.cumprod(sigmoids, dim=-1)
            
            dr_probs = torch.zeros_like(cum_probs)
            dr_probs[:, :-1] = cum_probs[:, :-1] - cum_probs[:, 1:]
            dr_probs[:, -1] = cum_probs[:, -1]
            dr_probs = dr_probs.squeeze(0)
            
            dr_class = int((corn_logits > 0).sum(dim=-1).item())
        else:
            dr_probs = torch.softmax(dr_logits, dim=-1).squeeze(0)
            dr_class = torch.argmax(dr_probs).item()

        dr_confidence = float(dr_probs[dr_class].item())
        dme_confidence = float(dme_probs[dme_class].item())

        # 5. Evidential subjective logic uncertainty evaluation
        dr_u = 0.0
        dme_u = 0.0
        referral_rec = False
        dr_evidence = []
        dme_evidence = []

        if self.config.heads.use_evidential and "dr_evidence" in outputs:
            dr_alpha = outputs["dr_evidence"]  # (1, K_dr)
            dme_alpha = outputs["dme_evidence"]  # (1, K_dme)
            
            dr_evidence = dr_alpha.squeeze(0).cpu().numpy().tolist()
            dme_evidence = dme_alpha.squeeze(0).cpu().numpy().tolist()

            dr_preds_tensor = torch.tensor([dr_class], device=self.device)
            dme_preds_tensor = torch.tensor([dme_class], device=self.device)

            referral_results = self.referral_module(
                dr_preds=dr_preds_tensor,
                dme_preds=dme_preds_tensor,
                dr_alpha=dr_alpha,
                dme_alpha=dme_alpha
            )

            dr_u = float(referral_results["dr_uncertainty"].item())
            dme_u = float(referral_results["dme_uncertainty"].item())
            referral_rec = bool(referral_results["referral_flag"].item())
        else:
            # Severity-only referral check fallback
            dr_sev_threshold = self.config.referral.dr_severity_threshold
            dme_sev_threshold = self.config.referral.dme_severity_threshold
            referral_rec = (dr_class >= dr_sev_threshold) or (dme_class >= dme_sev_threshold)

        # Standardized return structure
        return {
            "DR Grade": int(dr_class),
            "DME": int(dme_class),
            "Confidence": (dr_confidence + dme_confidence) / 2.0,
            "Uncertainty": (dr_u + dme_u) / 2.0,
            "Referral Decision": referral_rec,
            
            # Diagnostic details
            "DR Confidence": dr_confidence,
            "DME Confidence": dme_confidence,
            "DR Uncertainty": dr_u,
            "DME Uncertainty": dme_u,
            "Evidence DR": dr_evidence,
            "Evidence DME": dme_evidence
        }

    def predict_folder(
        self, 
        folder_path: str, 
        output_csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Scans a directory for images, runs batch predictions, and aggregates results.
        
        Args:
            folder_path (str): Target directory containing fundus images.
            output_csv_path (str, optional): Target file path to write results spreadsheet.

        Returns:
            pd.DataFrame: Tabular findings database.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder path not found: {folder_path}")

        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        image_files = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ])

        results_list = []
        for img_path in image_files:
            try:
                res = self.predict(img_path)
                res["image_filename"] = os.path.basename(img_path)
                res["image_path"] = img_path
                results_list.append(res)
            except Exception as e:
                print(f"Failed to process image: {img_path}. Error: {str(e)}")

        df_results = pd.DataFrame(results_list)

        # Re-order columns for clarity
        if not df_results.empty:
            main_cols = ["image_filename", "DR Grade", "DME", "Confidence", "Uncertainty", "Referral Decision"]
            remaining_cols = [c for c in df_results.columns if c not in main_cols]
            df_results = df_results[main_cols + remaining_cols]

            if output_csv_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
                df_results.to_csv(output_csv_path, index=False)
                print(f"Tabular predictions successfully written to: {output_csv_path}")

        return df_results
