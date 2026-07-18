import os
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import yaml


def resolve_kaggle_path(path: str) -> str:
    """Resolves local dataset paths to Kaggle dataset equivalents if running on Kaggle."""
    if not path:
        return path
        
    # Get project root (where config.py is located)
    project_root = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(project_root, path))
    if os.path.exists(abs_path):
        return abs_path
        
    norm_path = path.replace("\\", "/")
    
    # Kaggle input paths
    kaggle_aptos = "/kaggle/input/aptos-blindness-detection"
    kaggle_idrid = "/kaggle/input/idrid"
    
    # Check if running on Kaggle
    if os.path.exists("/kaggle/input") or os.path.exists("/kaggle/working"):
        # Map APTOS paths
        if "aptos" in norm_path.lower():
            if norm_path.endswith(".csv"):
                filename = os.path.basename(norm_path)
                candidate = os.path.join(kaggle_aptos, filename)
                if os.path.exists(candidate):
                    return candidate
            else:
                # Images folders
                if "val_images" in norm_path.lower():
                    candidate = os.path.join(kaggle_aptos, "train_images")
                elif "test_images" in norm_path.lower():
                    candidate = os.path.join(kaggle_aptos, "test_images")
                elif "train_images" in norm_path.lower():
                    candidate = os.path.join(kaggle_aptos, "train_images")
                else:
                    candidate = os.path.join(kaggle_aptos, os.path.basename(norm_path))
                if os.path.exists(candidate):
                    return candidate
                    
        # Map IDRiD paths
        elif "idrid" in norm_path.lower():
            if norm_path.endswith(".csv"):
                filename = os.path.basename(norm_path)
                candidate = os.path.join(kaggle_idrid, filename)
                if os.path.exists(candidate):
                    return candidate
            else:
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
                    
    return abs_path


@dataclass
class DatasetConfig:
    data_dir: str = "dataset/"
    train_csv: str = "dataset/integrated_train.csv"
    val_csv: str = "dataset/integrated_val.csv"
    test_csv: str = "dataset/integrated_test.csv"
    
    # Raw APTOS dataset paths
    aptos_train_csv: str = "dataset/aptos/train_1.csv"
    aptos_train_images: str = "dataset/aptos/train_images"
    aptos_val_csv: str = "dataset/aptos/valid.csv"
    aptos_val_images: str = "dataset/aptos/val_images"
    aptos_test_csv: str = "dataset/aptos/test.csv"
    aptos_test_images: str = "dataset/aptos/test_images"
    
    # Raw IDRiD dataset paths
    idrid_csv: str = "dataset/idrid/idrid_labels.csv"
    idrid_images: str = "dataset/idrid/Imagenes"
    
    # Dataloader configurations
    image_size: int = 512
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    augmentations: List[str] = field(
        default_factory=lambda: ["random_crop", "horizontal_flip", "vertical_flip", "color_jitter"]
    )

    def __post_init__(self):
        # Resolve all dataset paths
        self.data_dir = resolve_kaggle_path(self.data_dir)
        self.train_csv = resolve_kaggle_path(self.train_csv)
        self.val_csv = resolve_kaggle_path(self.val_csv)
        self.test_csv = resolve_kaggle_path(self.test_csv)
        
        self.aptos_train_csv = resolve_kaggle_path(self.aptos_train_csv)
        self.aptos_train_images = resolve_kaggle_path(self.aptos_train_images)
        self.aptos_val_csv = resolve_kaggle_path(self.aptos_val_csv)
        self.aptos_val_images = resolve_kaggle_path(self.aptos_val_images)
        self.aptos_test_csv = resolve_kaggle_path(self.aptos_test_csv)
        self.aptos_test_images = resolve_kaggle_path(self.aptos_test_images)
        
        self.idrid_csv = resolve_kaggle_path(self.idrid_csv)
        self.idrid_images = resolve_kaggle_path(self.idrid_images)


@dataclass
class EncoderConfig:
    name: str = "efficientnet_b4"  # Encoder name (e.g., "efficientnet_b4", "retfound")
    pretrained: bool = True
    checkpoint_path: Optional[str] = None  # Pre-trained encoder weights path
    freeze_features: bool = False
    dropout: float = 0.3                   # Default dropout rate


@dataclass
class HeadsConfig:
    dr_num_classes: int = 5   # Number of DR classes (0-4)
    dme_num_classes: int = 3  # Number of DME classes (0-2)
    use_evidential: bool = True
    projection_dim: int = 512 # Standardized projection dimension size


@dataclass
class LossConfig:
    ordinal_weight: float = 1.0
    evidential_weight: float = 0.5  # Evidence loss weight
    multitask_weights: Dict[str, float] = field(
        default_factory=lambda: {"dr": 1.0, "dme": 1.0}  # Multi-task loss weights
    )
    lambda1: float = 1.0            # Weight for DR task loss
    lambda2: float = 1.0            # Weight for DME task loss
    lambda3: float = 0.5            # Weight for Evidential loss terms
    dme_loss_type: str = "bce"      # DME loss type ('bce', 'focal', or 'ce')
    dme_class_weights: Optional[List[float]] = None  # Class-wise weighting for DME
    focal_gamma: float = 2.0        # Gamma exponent for focal loss term
    ordinal_method: str = "corn"    # Ordinal loss method ('corn' or 'emd')
    evidential_loss_type: str = "mse" # Evidential loss type ('mse' or 'ce')


@dataclass
class ReferralConfig:
    dr_severity_threshold: int = 2        # Refer to specialist if predicted DR >= Moderate
    dme_severity_threshold: int = 1       # Refer to specialist if predicted DME >= Mild
    uncertainty_threshold: float = 0.4     # Refer to specialist if epistemic uncertainty >= 0.4


@dataclass
class TrainerConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "AdamW"
    lr_scheduler: str = "CosineAnnealingLR"
    device: str = "cuda"
    seed: int = 42
    save_dir: str = "outputs/checkpoints/"
    checkpoint_path: Optional[str] = None     # Checkpoint paths (loading)
    log_dir: str = "outputs/logs/"
    project_name: str = "ORD-MED"
    experiment_name: str = "base_experiment"
    use_amp: bool = True                      # Mixed precision option (Automatic Mixed Precision)
    patience: int = 10                        # Early stopping patience epochs


@dataclass
class OutputsConfig:
    base_dir: str = "/kaggle/working/ORD-MED/outputs"
    checkpoints: Optional[str] = None
    logs: Optional[str] = None
    tensorboard: Optional[str] = None
    evaluation: Optional[str] = None
    gradcam: Optional[str] = None
    predictions: Optional[str] = None
    metrics: Optional[str] = None
    plots: Optional[str] = None

    def __post_init__(self):
        # Store what was explicitly set by the user/YAML to preserve overrides
        if not hasattr(self, "_explicit_checkpoints"):
            self._explicit_checkpoints = self.checkpoints
            self._explicit_logs = self.logs
            self._explicit_tensorboard = self.tensorboard
            self._explicit_evaluation = self.evaluation
            self._explicit_gradcam = self.gradcam
            self._explicit_predictions = self.predictions
            self._explicit_metrics = self.metrics
            self._explicit_plots = self.plots

        # Resolve paths: if a path was not explicitly set, derive it from the current base_dir
        self.checkpoints = self._explicit_checkpoints or os.environ.get("ORD_MED_SAVE_DIR", os.path.join(self.base_dir, "checkpoints"))
        self.logs = self._explicit_logs or os.environ.get("ORD_MED_LOG_DIR", os.path.join(self.base_dir, "logs"))
        self.tensorboard = self._explicit_tensorboard or os.path.join(self.base_dir, "tensorboard")
        self.evaluation = self._explicit_evaluation or os.path.join(self.base_dir, "evaluation")
        self.gradcam = self._explicit_gradcam or os.path.join(self.base_dir, "gradcam")
        self.predictions = self._explicit_predictions or os.path.join(self.base_dir, "predictions")
        self.metrics = self._explicit_metrics or os.path.join(self.base_dir, "metrics")
        self.plots = self._explicit_plots or os.path.join(self.base_dir, "plots")


@dataclass
class VisualizationConfig:
    use_gradcam: bool = True
    cam_target_layers: List[str] = field(default_factory=lambda: ["backbone.conv_head"])


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    referral: ReferralConfig = field(default_factory=ReferralConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        # Propagate resolved centralized outputs paths to trainer settings if defaults are used
        if self.trainer.save_dir in ("outputs/checkpoints/", "outputs/checkpoints"):
            self.trainer.save_dir = self.outputs.checkpoints
        if self.trainer.log_dir in ("outputs/logs/", "outputs/logs"):
            self.trainer.log_dir = self.outputs.logs

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Config":
        """Loads configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f) or {}

        # Resolve nested dicts to dataclasses
        dataset_cfg = DatasetConfig(**yaml_dict.get("dataset", {}))
        encoder_cfg = EncoderConfig(**yaml_dict.get("encoder", {}))
        heads_cfg = HeadsConfig(**yaml_dict.get("heads", {}))
        loss_cfg = LossConfig(**yaml_dict.get("loss", {}))
        referral_cfg = ReferralConfig(**yaml_dict.get("referral", {}))
        trainer_cfg = TrainerConfig(**yaml_dict.get("trainer", {}))
        outputs_cfg = OutputsConfig(**yaml_dict.get("outputs", {}))
        vis_cfg = VisualizationConfig(**yaml_dict.get("visualization", {}))

        return cls(
            dataset=dataset_cfg,
            encoder=encoder_cfg,
            heads=heads_cfg,
            loss=loss_cfg,
            referral=referral_cfg,
            trainer=trainer_cfg,
            outputs=outputs_cfg,
            visualization=vis_cfg
        )

    def save_to_yaml(self, yaml_path: str) -> None:
        """Saves configuration to a YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def parse_args() -> Config:
    """Parses command line arguments and returns a Config object."""
    parser = argparse.ArgumentParser(description="ORD-MED Centralized Configuration Parser")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    
    # Allow overriding critical settings directly
    parser.add_argument("--encoder", type=str, default=None, help="Override encoder (backbone) name")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override execution device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override model checkpoint load path")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision option (AMP)")
    
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = Config.load_from_yaml(args.config)
    else:
        config = Config()

    # Overrides
    if args.encoder is not None:
        config.encoder.name = args.encoder
    if args.lr is not None:
        config.trainer.lr = args.lr
    if args.epochs is not None:
        config.trainer.epochs = args.epochs
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
    if args.device is not None:
        config.trainer.device = args.device
    if args.checkpoint is not None:
        config.trainer.checkpoint_path = args.checkpoint
    if args.no_amp:
        config.trainer.use_amp = False

    return config


if __name__ == "__main__":
    # Test script utility
    cfg = Config()
    print("ORD-MED centralized configuration verified successfully.")
