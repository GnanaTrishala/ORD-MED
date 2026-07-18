import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple


def calculate_md5(filepath: str) -> str:
    """Calculates MD5 hash of a file to check for exact duplicate images."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_nested_image_dir(base_dir: str) -> str:
    """Detects if folder structures are nested and returns the final directory containing images."""
    name = os.path.basename(base_dir)
    nested = os.path.join(base_dir, name)
    if os.path.exists(nested) and os.path.isdir(nested):
        return nested
    # Walk to locate first directory containing image files
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            return root
    return base_dir


def process_dataset(
    csv_path: str,
    img_dir_base: str,
    dataset_name: str,
    is_idrid: bool = False
) -> pd.DataFrame:
    """
    Verifies images, verifies labels, detects corrupted files, 
    and checks for duplicates within the dataset.
    """
    print(f"\nProcessing {dataset_name} dataset...")
    df = pd.read_csv(csv_path)
    
    # Automatically resolve nested image folder
    img_dir = find_nested_image_dir(img_dir_base)
    print(f"Resolved image folder: {img_dir}")

    # Set up column mappings
    id_col = "id_code"
    dr_col = "diagnosis"
    dme_col = "Risk of macular edema " if is_idrid else None

    valid_records = []
    corrupted_count = 0
    invalid_label_count = 0
    duplicate_count = 0
    seen_hashes = set()

    for idx, row in df.iterrows():
        id_code = str(row[id_col]).strip()
        
        # Resolve file path (check for extensions)
        img_name = id_code
        if not (img_name.endswith(".png") or img_name.endswith(".jpg") or img_name.endswith(".jpeg")):
            # Check file system
            if os.path.exists(os.path.join(img_dir, img_name + ".png")):
                img_name += ".png"
            elif os.path.exists(os.path.join(img_dir, img_name + ".jpg")):
                img_name += ".jpg"
            elif os.path.exists(os.path.join(img_dir, img_name + ".jpeg")):
                img_name += ".jpeg"
                
        img_path = os.path.join(img_dir, img_name)

        # 1. Verify file exists
        if not os.path.exists(img_path):
            print(f"  [MISSING] Image not found: {img_path}")
            corrupted_count += 1
            continue

        # 2. Verify image integrity (not corrupted)
        try:
            with Image.open(img_path) as img:
                img.verify()  # Check file structure
            
            # Re-open to inspect size and load header
            with Image.open(img_path) as img:
                w, h = img.size
                # Force load to catch encoding errors
                img.load()
        except Exception as e:
            print(f"  [CORRUPTED] Failed to open image: {img_path}. Error: {str(e)}")
            corrupted_count += 1
            continue

        # 3. Verify labels validity
        dr_val = int(row[dr_col])
        if dr_val < 0 or dr_val > 4:
            print(f"  [INVALID LABEL] DR grade {dr_val} out of bounds: {img_path}")
            invalid_label_count += 1
            continue

        dme_val = -100  # Default missing label indicator
        if is_idrid and dme_col:
            dme_val = int(row[dme_col])
            if dme_val < 0 or dme_val > 2:
                print(f"  [INVALID LABEL] DME stage {dme_val} out of bounds: {img_path}")
                invalid_label_count += 1
                continue

        # 4. Detect exact duplicate images using MD5 hash
        try:
            file_hash = calculate_md5(img_path)
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(file_hash)
        except Exception:
            pass

        valid_records.append({
            "image_path": os.path.relpath(img_path, os.getcwd()).replace("\\", "/"),
            "dr_label": dr_val,
            "dme_label": dme_val,
            "dataset_source": dataset_name.lower()
        })

    print(f"Processed {len(df)} rows:")
    print(f"  - Valid records kept: {len(valid_records)}")
    print(f"  - Corrupted/Missing images: {corrupted_count}")
    print(f"  - Invalid labels: {invalid_label_count}")
    print(f"  - Duplicate images removed: {duplicate_count}")

    return pd.DataFrame(valid_records)


def generate_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Computes and prints database statistics."""
    print("\n=======================================================")
    print("INTEGRATED DATASET STATISTICS")
    print("=======================================================")
    
    splits = {"Train": train_df, "Validation": val_df, "Test": test_df}
    
    total_samples = 0
    for name, df in splits.items():
        print(f"\n--- {name} Split (Total: {len(df)} samples) ---")
        total_samples += len(df)
        
        # Sources count
        sources = df["dataset_source"].value_counts()
        for src, count in sources.items():
            print(f"  Source {src.upper()}: {count} samples")

        # DR distribution
        dr_dist = df[df["dr_label"] != -100]["dr_label"].value_counts().sort_index()
        print("  DR Grade Distribution:")
        for grade, count in dr_dist.items():
            print(f"    Grade {grade}: {count}")
            
        # DME distribution
        dme_dist = df[df["dme_label"] != -100]["dme_label"].value_counts().sort_index()
        if len(dme_dist) > 0:
            print("  DME Stage Distribution:")
            for stage, count in dme_dist.items():
                print(f"    Stage {stage}: {count}")
        else:
            print("  DME Stage Distribution: No DME annotations present in this split.")
            
    print(f"\nTotal integrated database size: {total_samples} samples.")
    print("=======================================================\n")


def plot_distributions(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, save_path: str):
    """Generates and saves the class distribution bar charts."""
    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Plot DR Distribution
    dr_data = df_all[df_all["dr_label"] != -100]
    dr_counts = dr_data.groupby(["dr_label", "dataset_source"]).size().unstack(fill_value=0)
    dr_counts.plot(kind="bar", stacked=True, color=["#1da1f2", "#7c5295"], ax=axes[0])
    axes[0].set_title("Diabetic Retinopathy (DR) Class Distribution", fontsize=11, fontweight="bold", pad=12)
    axes[0].set_xlabel("DR Severity Grade", fontweight="bold")
    axes[0].set_ylabel("Sample Count", fontweight="bold")
    axes[0].set_xticklabels([f"Grade {i}" for i in range(5)], rotation=0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].legend(title="Source")

    # Plot DME Distribution
    dme_data = df_all[df_all["dme_label"] != -100]
    if len(dme_data) > 0:
        dme_counts = dme_data.groupby(["dme_label", "dataset_source"]).size().unstack(fill_value=0)
        dme_counts.plot(kind="bar", stacked=True, color=["#1da1f2", "#7c5295"], ax=axes[1])
        axes[1].set_title("Diabetic Macular Edema (DME) Class Distribution", fontsize=11, fontweight="bold", pad=12)
        axes[1].set_xlabel("DME Severity Stage", fontweight="bold")
        axes[1].set_ylabel("Sample Count", fontweight="bold")
        axes[1].set_xticklabels([f"Stage {i}" for i in range(3)], rotation=0)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        axes[1].legend(title="Source")
    else:
        axes[1].text(0.5, 0.5, "No DME Labels Present", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution plot to: {save_path}")


def save_sample_grid(df: pd.DataFrame, save_path: str):
    """Draws and saves a grid of sample fundus images."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=300)
    
    # Pick a few random samples
    np.random.seed(42)
    samples = df.sample(min(8, len(df)), replace=False)
    
    for idx, (index, row) in enumerate(samples.iterrows()):
        r = idx // 4
        c = idx % 4
        
        ax = axes[r, c]
        img_path = row["image_path"]
        
        try:
            with Image.open(img_path) as img:
                # Resize for standard grid loading
                img_resized = img.resize((256, 256))
                ax.imshow(img_resized)
                
            source_name = row["dataset_source"].upper()
            dr_lbl = row["dr_label"]
            dme_lbl = "N/A" if row["dme_label"] == -100 else row["dme_label"]
            
            ax.set_title(f"{source_name}\nDR: {dr_lbl} | DME: {dme_lbl}", fontsize=8, pad=5)
        except Exception:
            ax.text(0.5, 0.5, "Error Loading", ha="center", va="center")
            
        ax.axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved sample image grid to: {save_path}")


def resolve_source_path(path: str) -> str:
    """Resolves a local dataset path to its Kaggle counterpart if on Kaggle/local missing."""
    if os.path.exists(path):
        return path
        
    norm_path = path.replace("\\", "/")
    
    # Kaggle input paths
    kaggle_aptos = "/kaggle/input/aptos-blindness-detection"
    kaggle_idrid = "/kaggle/input/idrid"
    
    if "dataset/aptos" in norm_path:
        if norm_path.endswith(".csv"):
            candidate = norm_path.replace("dataset/aptos", kaggle_aptos)
            if os.path.exists(candidate):
                return candidate
        else:
            if "val_images" in norm_path:
                candidate = os.path.join(kaggle_aptos, "train_images")
            elif "test_images" in norm_path:
                candidate = os.path.join(kaggle_aptos, "test_images")
            elif "train_images" in norm_path:
                candidate = os.path.join(kaggle_aptos, "train_images")
            else:
                candidate = norm_path.replace("dataset/aptos", kaggle_aptos)
            if os.path.exists(candidate):
                return candidate
                
    elif "dataset/idrid" in norm_path:
        if norm_path.endswith(".csv"):
            candidate = norm_path.replace("dataset/idrid", kaggle_idrid)
            if os.path.exists(candidate):
                return candidate
        else:
            if "Imagenes" in norm_path:
                candidates = [
                    os.path.join(kaggle_idrid, "Imagenes"),
                    os.path.join(kaggle_idrid, "Imagenes/Imagenes"),
                    os.path.join(kaggle_idrid, "disease-grading/disease-grading/Original Images/Training Set"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        return c
            candidate = norm_path.replace("dataset/idrid", kaggle_idrid)
            if os.path.exists(candidate):
                return candidate
                
    return path


def main():
    # 1. Process APTOS splits
    aptos_train = process_dataset(
        resolve_source_path("dataset/aptos/train_1.csv"),
        resolve_source_path("dataset/aptos/train_images"),
        "aptos", is_idrid=False
    )
    aptos_val = process_dataset(
        resolve_source_path("dataset/aptos/valid.csv"),
        resolve_source_path("dataset/aptos/val_images"),
        "aptos", is_idrid=False
    )
    aptos_test = process_dataset(
        resolve_source_path("dataset/aptos/test.csv"),
        resolve_source_path("dataset/aptos/test_images"),
        "aptos", is_idrid=False
    )

    # 2. Process IDRiD complete dataset
    idrid_all = process_dataset(
        resolve_source_path("dataset/idrid/idrid_labels.csv"),
        resolve_source_path("dataset/idrid/Imagenes"),
        "idrid", is_idrid=True
    )

    # 3. Automatically split IDRiD into train, validation, and test splits
    print("\nExecuting stratified split on IDRiD (80% train, 10% val, 10% test)...")
    idrid_train, idrid_temp = train_test_split(
        idrid_all,
        test_size=0.20,
        random_state=42,
        stratify=idrid_all["dr_label"]
    )
    idrid_val, idrid_test = train_test_split(
        idrid_temp,
        test_size=0.50,
        random_state=42,
        stratify=idrid_temp["dr_label"]
    )
    print(f"IDRiD split results: Train={len(idrid_train)}, Val={len(idrid_val)}, Test={len(idrid_test)}")

    # 4. Integrate both datasets
    train_integrated = pd.concat([aptos_train, idrid_train], ignore_index=True)
    val_integrated = pd.concat([aptos_val, idrid_val], ignore_index=True)
    test_integrated = pd.concat([aptos_test, idrid_test], ignore_index=True)

    # Shuffle training set for good measure
    train_integrated = train_integrated.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 5. Export integrated metadata CSV files
    os.makedirs("dataset", exist_ok=True)
    train_integrated.to_csv("dataset/integrated_train.csv", index=False)
    val_integrated.to_csv("dataset/integrated_val.csv", index=False)
    test_integrated.to_csv("dataset/integrated_test.csv", index=False)
    print("\nExported integrated metadata CSV files to dataset/integrated_*.csv")

    # 6. Generate statistics and figures
    generate_statistics(train_integrated, val_integrated, test_integrated)
    
    figures_dir = "outputs/figures"
    plot_distributions(train_integrated, val_integrated, test_integrated, os.path.join(figures_dir, "integrated_dataset_distribution.png"))
    save_sample_grid(train_integrated, os.path.join(figures_dir, "integrated_sample_images.png"))


if __name__ == "__main__":
    main()
