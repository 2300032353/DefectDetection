"""
Central configuration for the multi-category PatchCore anomaly detection system.
Supported categories: screw, bottle, metal_nut
"""

from pathlib import Path

# ============================================================
# PROJECT ROOT & PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\hp\Desktop\DefectDetection")
DATASET_ROOT = PROJECT_DIR / "archive (1)"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_REGISTRY_PATH = MODELS_DIR / "checkpoint_registry.json"
PROTOTYPES_PATH = MODELS_DIR / "prototypes.npz"

# ============================================================
# SUPPORTED CATEGORIES (MVTec AD)
# ============================================================

SUPPORTED_CATEGORIES = ["screw", "bottle", "metal_nut"]

CATEGORY_LABELS = {
    "screw": "Screw (Fastener)",
    "bottle": "Glass Bottle",
    "metal_nut": "Metal Hex Nut",
}

CATEGORY_DESCRIPTION = {
    "screw": "Detects defects: scratch_head, scratch_neck, manipulated_front, thread_side, thread_top",
    "bottle": "Detects defects: broken_large, broken_small, contamination",
    "metal_nut": "Detects defects: bent, color, flip, scratch",
}

# Dataset image counts (verified during inspection)
CATEGORY_DATASET_STATS = {
    "screw":     {"train_normal": 300, "test_normal": 41,  "test_defective": 120, "total_test": 161},
    "bottle":    {"train_normal": 209, "test_normal": 20,  "test_defective": 63,  "total_test": 83},
    "metal_nut": {"train_normal": 220, "test_normal": 22,  "test_defective": 93,  "total_test": 115},
}

# ============================================================
# PATCHCORE MODEL CONFIG
# ============================================================

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
PRE_TRAINED = True
CORESET_SAMPLING_RATIO = 0.1
IMAGE_SIZE = (256, 256)

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 0

# ============================================================
# UNSUPERVISED TRAINING ENFORCEMENT
# ============================================================
# Each PatchCore model is trained ONLY on train/good (normal) images.
# The anomalib MVTecAD datamodule enforces this by default for the
# train split — we double-check the dataset structure before training.

def validate_dataset_structure(category: str) -> dict:
    """
    Verify that a category dataset is correctly structured:
      - train/good/ exists (only normal images allowed for training)
      - test/ exists with good/ + defect_type/ subfolders
      - ground_truth/ exists with masks
    Returns a dict with counts or raises a descriptive error.
    """
    cat_dir = DATASET_ROOT / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(f"Category folder not found: {cat_dir}")

    train_good_dir = cat_dir / "train" / "good"
    if not train_good_dir.is_dir():
        raise FileNotFoundError(
            f"train/good/ not found for {category}. "
            f"Unsupervised PatchCore requires ONLY normal training images."
        )

    test_dir = cat_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test/ folder not found for {category}")

    test_good_dir = test_dir / "good"
    if not test_good_dir.is_dir():
        raise FileNotFoundError(f"test/good/ not found for {category}")

    gt_dir = cat_dir / "ground_truth"
    has_gt = gt_dir.is_dir()

    n_train_good = len(list(train_good_dir.glob("*.png")))
    n_test_good = len(list(test_good_dir.glob("*.png")))

    defect_types = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    ])
    n_test_defective = sum(
        len(list((test_dir / dt).glob("*.png"))) for dt in defect_types
    )

    return {
        "category": category,
        "n_train_good_only": n_train_good,  # ONLY normal training
        "n_test_good": n_test_good,
        "n_test_defective": n_test_defective,
        "defect_types": defect_types,
        "has_ground_truth_masks": has_gt,
    }
