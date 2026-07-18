from typing import Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config


def get_train_transforms(config: Config) -> Any:
    """
    Returns data augmentations and normalization for the training split.
    Uses clinical-grade fundus processing like CLAHE and slight color adjustments.
    """
    img_size = config.dataset.image_size
    
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # CLAHE is highly recommended for enhancing contrast in retinal fundus images
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        
        # ImageNet normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(config: Config) -> Any:
    """
    Returns preprocessing and normalization for validation/test splits.
    """
    img_size = config.dataset.image_size
    
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_inference_transforms(config: Config) -> Any:
    """
    Returns preprocessing for inference mode.
    """
    return get_val_transforms(config)
