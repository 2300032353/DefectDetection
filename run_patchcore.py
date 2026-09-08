"""
Train a PatchCore model for a specific MVTec AD category.
Each model is trained on ONLY train/good (normal) images -> unsupervised anomaly detection.

Usage:
  .venv\\Scripts\\python.exe run_patchcore.py --category screw
  .venv\\Scripts\\python.exe run_patchcore.py --category bottle
  .venv\\Scripts\\python.exe run_patchcore.py --category metal_nut
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

from config import (
    SUPPORTED_CATEGORIES,
    DATASET_ROOT,
    RESULTS_DIR,
    CHECKPOINT_REGISTRY_PATH,
    BACKBONE,
    LAYERS,
    PRE_TRAINED,
    CORESET_SAMPLING_RATIO,
    IMAGE_SIZE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    NUM_WORKERS,
    validate_dataset_structure,
    CATEGORY_LABELS,
)


def find_latest_checkpoint(category: str) -> Path | None:
    """
    After anomalib training, find the newest model.ckpt inside:
    results/Patchcore/MVTecAD/{category}/<version>/weights/lightning/model.ckpt
    """
    cat_root = RESULTS_DIR / "Patchcore" / "MVTecAD" / category
    if not cat_root.is_dir():
        return None

    candidates = sorted(
        cat_root.glob("*/weights/lightning/model.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def update_checkpoint_registry(category: str, ckpt_path: Path):
    """Update the checkpoint registry JSON with the latest model path."""
    if CHECKPOINT_REGISTRY_PATH.exists():
        with open(CHECKPOINT_REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {"categories": {}, "last_updated": None}

    if "categories" not in registry:
        registry["categories"] = {}

    registry["categories"][category] = {
        "checkpoint_path": str(ckpt_path.resolve()),
        "trained_on": datetime.now().isoformat(timespec="seconds"),
        "backbone": BACKBONE,
        "layers": LAYERS,
        "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
        "train_only_normal": True,
    }
    registry["last_updated"] = datetime.now().isoformat(timespec="seconds")

    with open(CHECKPOINT_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PatchCore on ONE MVTec AD category using only normal train/good images."
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=SUPPORTED_CATEGORIES,
        help=f"Category to train. Supported: {SUPPORTED_CATEGORIES}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    category = args.category

    sep = "=" * 65
    print("\n" + sep)
    print(f"  PATCHCORE UNSUPERVISED TRAINING  —  {category.upper()}")
    print(sep)

    # -------- 1. Validate dataset: ONLY normal images for training --------
    print("\n[1/5] Validating dataset structure ...")
    stats = validate_dataset_structure(category)
    print(f"   Category label           : {CATEGORY_LABELS[category]}")
    print(f"   train/good images (ONLY) : {stats['n_train_good_only']}")
    print(f"   test/good  images        : {stats['n_test_good']}")
    print(f"   test/defect images       : {stats['n_test_defective']}")
    print(f"   Defect types             : {stats['defect_types']}")
    print(f"   Ground truth masks exist : {stats['has_ground_truth_masks']}")
    print("   ⚠  Model will be trained on train/good NORMAL images ONLY — "
          "no defect images used in training.")

    # -------- 2. Build datamodule (anomalib enforces train split = train/good) --------
    print("\n[2/5] Building MVTecAD datamodule ...")
    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category=category,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    datamodule.setup()
    print(f"   Train dataset size (normal only): {len(datamodule.train_data)}")
    print(f"   Test  dataset size  (normal+def) : {len(datamodule.test_data)}")
    print(f"   Image size / batch size          : {IMAGE_SIZE} / {TRAIN_BATCH_SIZE}")

    # -------- 3. PatchCore model --------
    print("\n[3/5] Creating PatchCore model ...")
    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=PRE_TRAINED,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
    )
    print(f"   Backbone                : {BACKBONE}")
    print(f"   Feature layers          : {LAYERS}")
    print(f"   Coreset sampling ratio  : {CORESET_SAMPLING_RATIO}")
    print(f"   Pre-trained ImageNet    : {PRE_TRAINED}")

    # -------- 4. Train --------
    engine = Engine()
    print(f"\n[4/5] Starting training on {category} ...")
    print("   (Running on CPU — please be patient, typical time 5-15 minutes)")
    t0 = datetime.now()

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    train_time = datetime.now() - t0
    print(f"   Training finished. Elapsed: {train_time}")

    # -------- 5. Test & save checkpoint registry --------
    print("\n[5/5] Running test and saving checkpoint registry ...")
    results = engine.test(
        model=model,
        datamodule=datamodule,
    )
    print("\n>>>  ENGINE TEST METRICS  <<<")
    for r in results:
        for k, v in r.items():
            v_str = f"{float(v):.5f}" if isinstance(v, (int, float)) else str(v)
            print(f"   {k:35s} : {v_str}")

    ckpt = find_latest_checkpoint(category)
    if ckpt is not None:
        update_checkpoint_registry(category, ckpt)
        print(f"\n✅ Checkpoint saved to registry:")
        print(f"   {ckpt}")
        print(f"   Registry file: {CHECKPOINT_REGISTRY_PATH}")
    else:
        print("\n⚠  Warning: could not find the trained checkpoint file. "
              "Please check results/Patchcore/MVTecAD/{category}/ manually.")

    print("\n" + sep)
    print(f"  DONE — {category} PatchCore trained & evaluated.")
    print(sep + "\n")


if __name__ == "__main__":
    main()
