from pathlib import Path
import json

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
)


PROJECT_DIR = Path(r"C:\Users\hp\Desktop\DefectDetection")
CHECKPOINT_PATH = (
    PROJECT_DIR
    / "results" / "Patchcore" / "MVTecAD" / "screw" / "v3"
    / "weights" / "lightning" / "model.ckpt"
)
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def _to_py(v):
    """Convert tensor / numpy / numeric -> Python native."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


def run_full_evaluation():
    print("=" * 60)
    print("  COMPREHENSIVE EVALUATION — PATCHCORE SCREW")
    print("=" * 60)

    datamodule = MVTecAD(
        root=str(PROJECT_DIR / "archive (1)"),
        category="screw",
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
    )
    datamodule.setup()
    BATCH_SIZE = datamodule.eval_batch_size or 16
    print(f"   Train samples : {len(datamodule.train_data)}")
    print(f"   Test samples  : {len(datamodule.test_data)}")
    print(f"   Batch size    : {BATCH_SIZE}")

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
    )
    engine = Engine()

    # -------- 1. Engine-level test metrics --------
    print("\n[1/4] Running engine.test() ...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT_PATH),
    )
    print("\n>>>  ENGINE TEST METRICS  <<<")
    for r in test_results:
        for k in r.keys():
            v = r[k]
            v_py = _to_py(v)
            if isinstance(v_py, float):
                print(f"   {k:32s} : {v_py:.5f}")
            else:
                print(f"   {k:32s} : {v_py}")

    # -------- 2. Per-image prediction collection --------
    print("\n[2/4] Collecting per-image scores (predict on full test set) ...")
    test_loader = datamodule.test_dataloader()
    predictions = engine.predict(
        model=model,
        dataloaders=test_loader,
        ckpt_path=str(CHECKPOINT_PATH),
        return_predictions=True,
    )
    print(f"   Number of output batches: {len(predictions)}")

    y_true = []       # 0 normal, 1 defect
    y_scores = []     # anomaly score
    y_pred = []       # predicted label (0/1)
    defects = []      # defect type as string
    image_path_list = []
    anomaly_map_list = []

    for batch_idx, batch in enumerate(test_loader):
        pred = predictions[batch_idx]
        n = len(batch.image_path)
        for i in range(n):
            image_path = Path(batch.image_path[i])
            gt_label = int(batch.gt_label[i].cpu().item())
            defect_type = image_path.parent.name

            score = float(pred.pred_score[i].cpu().item())
            plabel = int(pred.pred_label[i].cpu().item())
            amap = np.array(pred.anomaly_map[i].cpu().squeeze().numpy())

            y_true.append(gt_label)
            y_scores.append(score)
            y_pred.append(plabel)
            defects.append(defect_type)
            image_path_list.append(image_path)
            anomaly_map_list.append(amap)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    print(f"   Images processed: {len(y_true)}")
    print(f"   Normal samples   : {(y_true == 0).sum()}")
    print(f"   Defective samples: {(y_true == 1).sum()}")

    # -------- 3. Metrics breakdown by defect type --------
    print("\n[3/4] Per-defect-type breakdown ...")
    types = sorted(set(defects))
    breakdown_rows = []
    print(f"   {'Type':<22s} {'N':>4s}  {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s}  {'Acc':>6s}  {'F1':>6s}")
    for t in types:
        idx = np.array([d == t for d in defects])
        tt = y_true[idx]
        pp = y_pred[idx]
        tp = int(((tt == 1) & (pp == 1)).sum())
        fn = int(((tt == 1) & (pp == 0)).sum())
        fp = int(((tt == 0) & (pp == 1)).sum())
        tn = int(((tt == 0) & (pp == 0)).sum())
        n = len(tt)
        acc = float((tt == pp).mean())
        f1 = float(f1_score(tt, pp, zero_division=0))
        print(f"   {t:<22s} {n:>4d}  {tp:>4d} {fp:>4d} {fn:>4d} {tn:>4d}  {acc:>6.3f}  {f1:>6.3f}")
        breakdown_rows.append({
            "defect_type": t, "n": n, "TP": tp, "FP": fp,
            "FN": fn, "TN": tn, "accuracy": acc, "f1": f1,
        })

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n>>>  OVERALL CONFUSION MATRIX  <<<")
    print(f"                    Pred NORMAL   Pred DEFECTIVE")
    print(f"   Actual NORMAL     {cm[0,0]:>9d}     {cm[0,1]:>9d}")
    print(f"   Actual DEFECTIVE  {cm[1,0]:>9d}     {cm[1,1]:>9d}")

    print("\n>>>  CLASSIFICATION REPORT  <<<")
    print(classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "DEFECTIVE"],
        digits=4, zero_division=0,
    ))

    # -------- 4. ROC & PR curves --------
    print("\n[4/4] Generating ROC / PR curves & plots ...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    scores_normal = y_scores[y_true == 0]
    scores_defect = y_scores[y_true == 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(fpr, tpr, lw=2, label=f"AUROC = {roc_auc:.4f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 0].set_title("ROC Curve (Image-level)", fontweight="bold")
    axes[0, 0].set_xlabel("False Positive Rate"); axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(recall, precision, lw=2, color="orange", label=f"AUPRC = {pr_auc:.4f}")
    axes[0, 1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(scores_normal, bins=30, alpha=0.7,
                    label=f"Normal (n={len(scores_normal)})", color="green", density=True)
    axes[1, 0].hist(scores_defect, bins=30, alpha=0.7,
                    label=f"Defective (n={len(scores_defect)})", color="red", density=True)
    axes[1, 0].set_title("Anomaly Score Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Anomaly Score"); axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    names = [r["defect_type"] for r in breakdown_rows]
    vals  = [r["accuracy"] for r in breakdown_rows]
    colors = ["green" if n == "good" else "salmon" for n in names]
    axes[1, 1].bar(names, vals, color=colors, edgecolor="black")
    axes[1, 1].set_title("Accuracy by Category", fontweight="bold")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        axes[1, 1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    plt.setp(axes[1, 1].get_xticklabels(), rotation=25, ha="right")

    fig.suptitle(
        f"PatchCore Screw — Image-level (AUROC {roc_auc:.4f} / F1 {f1_score(y_true, y_pred):.4f})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plots_path = REPORT_DIR / "01_metrics_dashboard.png"
    fig.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved metrics dashboard -> {plots_path}")

    # -------- 5. Qualitative examples --------
    print("\n[Bonus] Generating qualitative heatmap examples ...")
    rng = np.random.default_rng(42)

    def pick_examples(mask, k):
        cand = np.where(mask)[0]
        return cand[rng.choice(len(cand), min(k, len(cand)), replace=False)]

    normal_idx = pick_examples(np.array([d == "good" for d in defects]), 4)
    defect_idx = pick_examples(np.array([d != "good" for d in defects]), 6)
    all_idx = np.concatenate([normal_idx, defect_idx])

    all_image_paths = [image_path_list[i] for i in all_idx]
    all_scores = [y_scores[i] for i in all_idx]
    all_gt = [y_true[i] for i in all_idx]
    all_pl = [y_pred[i] for i in all_idx]
    all_maps = [anomaly_map_list[i] for i in all_idx]

    ncol = 5
    nrow = int(np.ceil(len(all_image_paths) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    for idx, (ipath, score, gtl, pl, amap) in enumerate(
        zip(all_image_paths, all_scores, all_gt, all_pl, all_maps)
    ):
        ax = axes[idx // ncol, idx % ncol]
        orig = np.array(Image.open(ipath).convert("RGB"))
        h, w = orig.shape[:2]
        amap_rz = np.array(Image.fromarray(amap).resize((w, h), Image.BILINEAR))
        amap_n = (amap_rz - amap_rz.min()) / (amap_rz.max() - amap_rz.min() + 1e-8)
        hm = (cm.jet(amap_n)[..., :3] * 255).astype(np.uint8)
        over = (0.55 * orig + 0.45 * hm).astype(np.uint8)
        ax.imshow(over)
        gt = "NORMAL" if gtl == 0 else f"DEFECT ({ipath.parent.name})"
        pred = "NORMAL" if pl == 0 else "DEFECT"
        c = "green" if gtl == pl else "red"
        ax.set_title(f"GT: {gt}\nPred: {pred}  s={score:.3f}", fontsize=9, color=c)
        ax.axis("off")
    for idx in range(len(all_image_paths), nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    fig.suptitle("PatchCore Screw — Qualitative Heatmap Examples", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ex_path = REPORT_DIR / "02_heatmap_examples.png"
    fig.savefig(ex_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved heatmap examples    -> {ex_path}")

    # -------- 6. Save JSON summary --------
    engine_test_flat = []
    for r in test_results:
        engine_test_flat.append({k: _to_py(r[k]) for k in r.keys()})

    summary = {
        "dataset": "MVTecAD / screw",
        "model": "PatchCore (wide_resnet50_2, layers2-3, coreset 0.1)",
        "n_train_normal": len(datamodule.train_data),
        "n_test_normal": int((y_true == 0).sum()),
        "n_test_defective": int((y_true == 1).sum()),
        "image_auroc": float(roc_auc),
        "image_auprc": float(pr_auc),
        "image_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "breakdown": breakdown_rows,
        "engine_test_results": engine_test_flat,
    }
    with open(REPORT_DIR / "03_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved JSON summary        -> {REPORT_DIR / '03_summary.json'}")

    print("\n" + "=" * 60)
    print("  DONE — all reports saved to ./reports/")
    print("=" * 60)


if __name__ == "__main__":
    run_full_evaluation()
