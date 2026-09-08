from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# =========================
# PROJECT PATHS
# =========================

PROJECT_DIR = Path(r"C:\Users\hp\Desktop\DefectDetection")

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "results"
    / "Patchcore"
    / "MVTecAD"
    / "screw"
    / "v3"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


# =========================
# HELPER: Resolve image path
# =========================

def resolve_image_path(user_input: str) -> Path | None:
    """
    Accepts:
      1. Full absolute path  (e.g. C:\\images\\my_screw.jpg)
      2. Relative path       (e.g. test.jpg, archive (1)\\screw\\test\\good\\000.png)
      3. Filename only       (e.g. 000.png) -> searches common locations
      4. Just a number       (e.g. 000) -> searches common test folders
    """
    user_input = user_input.strip().strip('"').strip("'")
    if not user_input:
        return None

    p = Path(user_input)

    # 1 & 2: Direct path (absolute or relative)
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
        return p.resolve()

    # 3: Filename only — search common folders
    search_dirs = [
        PROJECT_DIR,
        PROJECT_DIR / "archive (1)" / "screw" / "test" / "good",
        PROJECT_DIR / "archive (1)" / "screw" / "train" / "Bad",
        PROJECT_DIR / "archive (1)" / "screw" / "test" / "scratch_head",
        PROJECT_DIR / "archive (1)" / "screw" / "test" / "scratch_neck",
        PROJECT_DIR / "archive (1)" / "screw" / "test" / "manipulated_front",
        PROJECT_DIR / "archive (1)" / "screw" / "test" / "thread_side",
        PROJECT_DIR / "archive (1)" / "screw" / "test" / "thread_top",
        PROJECT_DIR / "results" / "Patchcore" / "latest" / "images" / "Bad",
    ]

    # Accept both common forms, e.g. "image1" and "image-1".
    stems = [user_input]
    if user_input.startswith("image") and user_input[5:].isdigit():
        stems.append(f"image-{user_input[5:]}")

    for d in search_dirs:
        if not d.is_dir():
            continue
        if "." in user_input:
            candidate = d / user_input
            if candidate.is_file() and candidate.suffix.lower() in VALID_EXTENSIONS:
                return candidate.resolve()
        else:
            for stem in stems:
                for ext in VALID_EXTENSIONS:
                    candidate = d / (stem + ext)
                    if candidate.is_file():
                        return candidate.resolve()

    return None


# =========================
# HELPER: Save visualisation
# =========================

def save_visualization(
    original_img_path: Path,
    anomaly_map: np.ndarray,
    pred_score: float,
    pred_label: int,
) -> Path:
    """
    Create a 3-panel figure:
      [Original Image]  [Anomaly Heatmap]  [Overlay]
    Saves to outputs/ folder. Returns saved path.
    """
    orig = Image.open(original_img_path).convert("RGB")
    orig_np = np.array(orig)

    h, w = orig_np.shape[:2]

    # Resize anomaly map to match original
    am = Image.fromarray(anomaly_map)
    am_resized = np.array(am.resize((w, h), resample=Image.BILINEAR))

    # Normalize for display
    am_norm = (am_resized - am_resized.min()) / (am_resized.max() - am_resized.min() + 1e-8)

    heatmap = cm.jet(am_norm)[..., :3]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    overlay = (0.6 * orig_np + 0.4 * heatmap_uint8).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_text = "DEFECTIVE" if pred_label == 1 else "NORMAL"
    color = "red" if pred_label == 1 else "green"

    axes[0].imshow(orig_np)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap_uint8)
    axes[1].set_title("Anomaly Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {label_text}  |  Anomaly Score: {pred_score:.4f}",
        fontsize=14, fontweight="bold", color=color,
    )
    fig.tight_layout()

    stem = original_img_path.stem
    out_path = OUTPUT_DIR / f"{stem}_result.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =========================
# MAIN
# =========================

def main():
    print("\n" + "=" * 55)
    print("      🔧 AI VISUAL DEFECT DETECTION SYSTEM 🔧")
    print("         PatchCore — Screw Quality Control")
    print("=" * 55)

    print("\nEnter the image path, file name, or image number.")
    print('Examples:')
    print('  • Full path:    "C:\\Users\\hp\\Pictures\\my_screw.jpg"')
    print('  • File name:    "000.png", "image-1.jpg", or "test_screw.jpg"')
    print('  • Number only:  "000"  (looks up in dataset)')
    print('  • Bad images:   "image1" or "image-1" (looks up in train/Bad)')
    print('  • Type "quit" to exit.\n')

    while True:
        user_input = input("➡  Enter image (or 'quit'): ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("👋 Goodbye.")
            break

        image_path = resolve_image_path(user_input)
        if image_path is None:
            print("❌ Could not find image. Please try again.\n")
            continue

        print(f"\n✅ Image found : {image_path.name}")
        print(f"   Path        : {image_path}")
        print("   Analyzing ..........")

        # ---- Model + Engine ----
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True,
            coreset_sampling_ratio=0.1,
        )

        dataset = PredictDataset(path=image_path, image_size=(256, 256))
        engine = Engine()

        try:
            predictions = engine.predict(
                model=model,
                dataset=dataset,
                ckpt_path=str(CHECKPOINT_PATH),
                return_predictions=True,
            )
        except Exception as e:
            print(f"❌ Inference error: {e}")
            continue

        if predictions is None or len(predictions) == 0:
            print("❌ No prediction returned.\n")
            continue

        for pred in predictions:
            # pred is a batch (even for single-image PredictDataset)
            # pred_score shape: torch.Size([N]); anomaly_map shape: [N, H, W]
            N = (
                len(pred.pred_score)
                if hasattr(pred.pred_score, "__len__")
                else 1
            )
            for i in range(N):
                score_t = pred.pred_score[i] if N > 1 else pred.pred_score
                label_t = pred.pred_label[i] if N > 1 else pred.pred_label
                score = float(score_t.cpu().item() if hasattr(score_t, "cpu") else score_t)
                label = int(label_t.cpu().item() if hasattr(label_t, "cpu") else label_t)

                am = pred.anomaly_map[i] if N > 1 else pred.anomaly_map
                am_arr = np.array(am.cpu() if hasattr(am, "cpu") else am)
                if am_arr.ndim == 3 and am_arr.shape[0] == 1:
                    am_arr = am_arr[0]

                print("\n" + "-" * 55)
                print("              📋 DETECTION RESULT")
                print("-" * 55)
                if label == 1:
                    print("   Prediction  :  🔴  DEFECTIVE  🔴")
                else:
                    print("   Prediction  :  🟢  NORMAL  🟢")
                print(f"   Anomaly Score : {score:.4f}")

                viz_path = save_visualization(image_path, am_arr, score, label)
                print(f"   Heatmap saved : {viz_path}")
                print("-" * 55 + "\n")

                input("Press Enter to continue...")


if __name__ == "__main__":
    main()