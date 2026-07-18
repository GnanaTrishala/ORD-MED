import os
from typing import Tuple, List, Dict, Any, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from datasets.transforms import get_train_transforms, get_val_transforms


def resolve_csv_path(path: str) -> str:
    """Resolves local CSV paths to Kaggle dataset equivalents if running on Kaggle."""
    if not path:
        return path
    if os.path.exists(path):
        return path
        
    norm_path = path.replace("\\", "/")
    filename = os.path.basename(norm_path)
    
    # Kaggle input paths
    kaggle_aptos = "/kaggle/input/aptos-blindness-detection"
    kaggle_idrid = "/kaggle/input/idrid"
    
    candidates = [
        path,
        os.path.join("dataset", filename),
    ]
    
    if "aptos" in norm_path.lower():
        candidates.append(os.path.join(kaggle_aptos, filename))
    elif "idrid" in norm_path.lower():
        candidates.append(os.path.join(kaggle_idrid, filename))
        
    for c in candidates:
        if os.path.exists(c):
            return c
            
    return path


def resolve_dir_path(path: str) -> str:
    """Resolves local dataset directory paths to Kaggle dataset input locations."""
    if not path:
        return path
    if os.path.exists(path):
        return path
        
    norm_path = path.replace("\\", "/")
    
    # Kaggle input paths
    kaggle_aptos = "/kaggle/input/aptos-blindness-detection"
    kaggle_idrid = "/kaggle/input/idrid"
    
    if "aptos" in norm_path.lower():
        if "train_images" in norm_path:
            candidate = os.path.join(kaggle_aptos, "train_images")
        elif "test_images" in norm_path:
            candidate = os.path.join(kaggle_aptos, "test_images")
        elif "val_images" in norm_path:
            candidate = os.path.join(kaggle_aptos, "train_images")  # Map val to train on Kaggle
        else:
            candidate = kaggle_aptos
        if os.path.exists(candidate):
            return candidate
            
    elif "idrid" in norm_path.lower():
        if "imagenes" in norm_path.lower():
            candidates = [
                os.path.join(kaggle_idrid, "Imagenes"),
                os.path.join(kaggle_idrid, "Imagenes/Imagenes"),
                os.path.join(kaggle_idrid, "disease-grading/disease-grading/Original Images/Training Set"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
        candidate = os.path.join(kaggle_idrid, os.path.basename(norm_path))
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(kaggle_idrid):
            return kaggle_idrid
            
    return path


def resolve_image_path(path: str, data_dir: str = "dataset/") -> str:
    """
    Dynamically resolves image paths to handle both local development
    and Kaggle environment dataset paths (single/double nested).
    """
    if not path:
        return path
    if os.path.exists(path):
        return path

    norm_path = path.replace("\\", "/")
    filename = os.path.basename(norm_path)
    
    # Kaggle dataset base paths
    kaggle_aptos = "/kaggle/input/aptos-blindness-detection"
    kaggle_idrid = "/kaggle/input/idrid"
    
    # Handle APTOS
    if "aptos" in norm_path.lower():
        subfolder = "train_images"
        if "test" in norm_path.lower():
            subfolder = "test_images"
        elif "val" in norm_path.lower():
            subfolder = "train_images"
            
        candidates = [
            os.path.join(kaggle_aptos, subfolder, filename),
            os.path.join(kaggle_aptos, subfolder, subfolder, filename),
            os.path.join(data_dir, "aptos", subfolder, filename),
            os.path.join(data_dir, "aptos", subfolder, subfolder, filename),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

    # Handle IDRiD
    elif "idrid" in norm_path.lower():
        candidates = [
            os.path.join(kaggle_idrid, "Imagenes", filename),
            os.path.join(kaggle_idrid, "Imagenes/Imagenes", filename),
            os.path.join(kaggle_idrid, "disease-grading/disease-grading/Original Images/Training Set", filename),
            os.path.join(kaggle_idrid, "disease-grading/disease-grading/Original Images/Testing Set", filename),
            os.path.join(data_dir, "idrid/Imagenes", filename),
            os.path.join(data_dir, "idrid/Imagenes/Imagenes", filename),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
                
    return path


class DiabeticEyeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading eye fundus images and their associated
    Diabetic Retinopathy (DR) grades and Diabetic Macular Edema (DME) stages.
    """
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Any = None,
        is_training: bool = True
    ):
        """
        Args:
            csv_path (str): Path to CSV file containing columns like 'image_id', 'dr_grade', 'dme_stage'.
            img_dir (str): Directory where retinal images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_training (bool): If True, requires labels. If False (inference), works with images only.
        """
        self.csv_path = resolve_csv_path(csv_path)
        self.img_dir = resolve_dir_path(img_dir)
        self.transform = transform
        self.is_training = is_training

        # Load data annotations
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Annotation file not found: {csv_path} (resolved to {self.csv_path})")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Check columns to support both integrated and legacy formats
        if "image_path" in self.df.columns:
            self.use_integrated = True
            self.image_cols = "image_path"
            self.dr_col = "dr_label"
            self.dme_col = "dme_label"
        else:
            self.use_integrated = False
            self.image_cols = "image_id"
            self.dr_col = "dr_grade"
            self.dme_col = "dme_stage"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        
        if self.use_integrated:
            img_path = row[self.image_cols]
        else:
            img_name = row[self.image_cols]
            img_path = os.path.join(self.img_dir, img_name)

        # Resolve image path dynamically (e.g. locally or on Kaggle)
        img_path = resolve_image_path(img_path, data_dir=self.img_dir)

        # Load image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at path: {img_path}. Error: {str(e)}")

        # Convert to numpy array for Albumentations compatibility
        image = np.array(image)

        # Apply transforms (Augmentation or normalization)
        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            # Default fallback: transpose and scale
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        sample = {
            "image": image_tensor,
            "image_path": img_path
        }

        # Retrieve labels if in training/validation mode
        if self.is_training:
            dr_label = int(row[self.dr_col])
            dme_label = int(row[self.dme_col])
            sample["dr_label"] = torch.tensor(dr_label, dtype=torch.long)
            sample["dme_label"] = torch.tensor(dme_label, dtype=torch.long)

        return sample


def get_dataloaders(
    config: Config, 
    splits: List[str] = ["train", "val"]
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Constructs and returns PyTorch Dataloaders for train, val, or test sets.

    Args:
        config (Config): Configuration object.
        splits (list of str): List of splits to return, e.g. ["train", "val"] or ["test"].

    Returns:
        DataLoader or Tuple[DataLoader, DataLoader]: Returns dataloaders for the specified splits.
    """
    loaders = []

    for split in splits:
        if split == "train":
            transform = get_train_transforms(config)
            dataset = DiabeticEyeDataset(
                csv_path=config.dataset.train_csv,
                img_dir=config.dataset.data_dir,
                transform=transform,
                is_training=True
            )
            loader = DataLoader(
                dataset,
                batch_size=config.dataset.batch_size,
                shuffle=True,
                num_workers=config.dataset.num_workers,
                pin_memory=config.dataset.pin_memory,
                drop_last=True
            )
            loaders.append(loader)
            
        elif split == "val":
            transform = get_val_transforms(config)
            dataset = DiabeticEyeDataset(
                csv_path=config.dataset.val_csv,
                img_dir=config.dataset.data_dir,
                transform=transform,
                is_training=True
            )
            loader = DataLoader(
                dataset,
                batch_size=config.dataset.batch_size,
                shuffle=False,
                num_workers=config.dataset.num_workers,
                pin_memory=config.dataset.pin_memory,
                drop_last=False
            )
            loaders.append(loader)
            
        elif split == "test":
            transform = get_val_transforms(config)
            dataset = DiabeticEyeDataset(
                csv_path=config.dataset.test_csv,
                img_dir=config.dataset.data_dir,
                transform=transform,
                is_training=True  # Usually test sets have labels for evaluation
            )
            loader = DataLoader(
                dataset,
                batch_size=config.dataset.batch_size,
                shuffle=False,
                num_workers=config.dataset.num_workers,
                pin_memory=config.dataset.pin_memory,
                drop_last=False
            )
            loaders.append(loader)
            
        else:
            raise ValueError(f"Unknown split: {split}")

    return tuple(loaders) if len(loaders) > 1 else loaders[0]


def verify_and_report_dataset(config: Config, logger: Any) -> None:
    """
    Scans the configuration dataset splits, verifies image counts,
    label counts, missing files, and class distributions, printing
    a detailed clinical database report.
    """
    logger.info("=======================================================")
    logger.info("ORD-MED DATASET VERIFICATION REPORT")
    logger.info("=======================================================")
    
    splits_to_check = {
        "Train": config.dataset.train_csv,
        "Validation": config.dataset.val_csv,
        "Test": config.dataset.test_csv
    }
    
    for split_name, orig_csv_path in splits_to_check.items():
        csv_path = resolve_csv_path(orig_csv_path)
        logger.info(f"Resolving {split_name} Split:")
        logger.info(f"  Original CSV Path: {orig_csv_path}")
        logger.info(f"  Resolved CSV Path: {csv_path}")
        
        if not csv_path or not os.path.exists(csv_path):
            logger.warning(f"  - STATUS: NOT FOUND! Skipping verification.")
            continue
        logger.info(f"  - STATUS: FOUND!")
            
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV at {csv_path}. Error: {str(e)}")
            continue
            
        total_records = len(df)
        missing_files = []
        found_files_count = 0
        
        # Check columns
        if "image_path" in df.columns:
            image_col = "image_path"
            dr_col = "dr_label"
            dme_col = "dme_label"
            use_rel = True
        else:
            image_col = "image_id"
            dr_col = "dr_grade"
            dme_col = "dme_stage"
            use_rel = False
            
        # Get resolved image folder path
        resolved_img_dir = resolve_dir_path(config.dataset.data_dir)
        logger.info(f"  Original Image Dir: {config.dataset.data_dir}")
        logger.info(f"  Resolved Image Dir: {resolved_img_dir}")
            
        for idx, row in df.iterrows():
            img_name = str(row[image_col]).strip()
            if use_rel:
                img_path = img_name
            else:
                img_path = os.path.join(config.dataset.data_dir, img_name)
                
            # Resolve image path dynamically
            resolved_img_path = resolve_image_path(img_path, data_dir=resolved_img_dir)
            
            # Print resolution of the first image
            if idx == 0:
                logger.info(f"  - Sample Image Path Resolution:")
                logger.info(f"    Original: {img_path}")
                logger.info(f"    Resolved: {resolved_img_path}")
                
            if os.path.exists(resolved_img_path):
                found_files_count += 1
            else:
                missing_files.append(resolved_img_path)
                
        logger.info(f"  - Total records (label count): {total_records}")
        logger.info(f"  - Verified images found on disk: {found_files_count}")
        
        if len(missing_files) > 0:
            logger.error(f"  - WARNING: {len(missing_files)} missing image files!")
            for m in missing_files[:5]:
                logger.error(f"    Missing (resolved path): {m}")
        else:
            logger.info("  - Image file integrity: 100% complete (0 missing files)")
            
        # Source dataset count
        if "dataset_source" in df.columns:
            sources = df["dataset_source"].value_counts()
            for src, count in sources.items():
                logger.info(f"    Source {src.upper()}: {count} samples")
                
        # DR Class distribution
        if dr_col in df.columns:
            dr_dist = df[df[dr_col] != -100][dr_col].value_counts().sort_index()
            dr_str = ", ".join([f"Grade {k}: {v}" for k, v in dr_dist.items()])
            logger.info(f"    DR Class distribution:  {dr_str}")
            
        # DME Class distribution
        if dme_col in df.columns:
            dme_dist = df[df[dme_col] != -100][dme_col].value_counts().sort_index()
            if len(dme_dist) > 0:
                dme_str = ", ".join([f"Stage {k}: {v}" for k, v in dme_dist.items()])
                logger.info(f"    DME Class distribution: {dme_str}")
            else:
                logger.info("    DME Class distribution: None (masked or missing)")
                
    logger.info("=======================================================\n")

