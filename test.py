import argparse
import csv
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from infer import load_model, run_inference, InfDataset


def _global_scale(results, ref_size=224, margin=0.35):
    """Compute a uniform scale factor so that all predictions fit within
    `margin * ref_size` pixels of the image center."""
    vals = np.array([r["pred"] for r in results])  # [N, 2]
    max_abs = np.abs(vals).max()
    if max_abs < 1e-8:
        return 1.0
    return ref_size * margin / max_abs


def draw_prediction(img_path, pred, save_path, scale, ref_size=224):
    """Draw predicted (speed, angle) as a point on the original image.

    Mapping: angle -> x (positive = right), speed -> y (positive = forward = up).
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: cannot read {img_path}, skipping")
        return
    h, w = img.shape[:2]

    speed, angle = pred
    cx, cy = w // 2, h // 2

    # Scale proportionally to actual image size
    img_scale = scale * (min(h, w) / ref_size)
    px = int(cx + angle * img_scale)  # angle → right +
    py = int(cy - speed * img_scale)  # speed forward → up (y-)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    # — center reference (green +)
    cv2.drawMarker(img, (cx, cy), (60, 200, 60), cv2.MARKER_CROSS, 14, 1)

    # — prediction point (red cross + filled circle)
    cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.circle(img, (px, py), 5, (0, 0, 255), -1)

    # — direction arrow
    cv2.arrowedLine(img, (cx, cy), (px, py), (255, 180, 0), 1, tipLength=0.08)

    # — text overlay
    txt = f"Speed={speed:.3f}  Angle={angle:.3f}"
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3)
    cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    cv2.imwrite(save_path, img)


def build_composite(results, save_path, scale, ref_size=224,
                    grid_size=(4, 4), figscale=4):
    """Arrange random samples into a grid figure and save."""
    n_cells = grid_size[0] * grid_size[1]
    random.seed(42)
    if len(results) < n_cells:
        samples = results
    else:
        samples = random.sample(results, n_cells)

    fig, axes = plt.subplots(*grid_size, figsize=(grid_size[1] * figscale,
                                                   grid_size[0] * figscale))
    axes = axes.flatten() if n_cells > 1 else [axes]

    for ax, sample in zip(axes, samples):
        img = cv2.imread(sample["path"])
        if img is None:
            ax.text(0.5, 0.5, "No image", ha="center", va="center")
            ax.axis("off")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        speed, angle = sample["pred"]

        img_scale = scale * (min(h, w) / ref_size)
        cx, cy = w // 2, h // 2
        px = int(cx + angle * img_scale)
        py = int(cy - speed * img_scale)

        ax.imshow(img_rgb)
        ax.plot(cx, cy, "g+", markersize=10, mew=1.5)
        ax.plot(px, py, "ro", markersize=7, fillstyle="full", mfc="red")
        ax.annotate("", xy=(px, py), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="orange", lw=1.2))
        ax.set_title(f"Speed={speed:.3f}  Angle={angle:.3f}", fontsize=7)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("LSNet Inference & Visualization")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("-d", "--data-dir", type=str, required=True,
                        help="Directory with images/ and radars/ subfolders")
    parser.add_argument("-s", "--save-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--radar-length", type=int, default=360)
    parser.add_argument("--grid", type=int, nargs=2, default=(4, 4),
                        help="Grid rows cols for composite image")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Model
    print(f"[1/5] Loading model from {args.model} …")
    model = load_model(args.model, args.device, args.radar_length)
    print(f"       Model loaded on {args.device}")

    # 2. Dataset & DataLoader
    print(f"[2/5] Loading data from {args.data_dir} …")
    dataset = InfDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch,
                        shuffle=False, num_workers=args.num_workers)
    print(f"       {len(dataset)} samples")

    # 3. Inference
    print("[3/5] Running inference …")
    results = run_inference(model, loader, args.device)

    # 4. Save CSV
    csv_path = os.path.join(args.save_dir, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "speed", "angle"])
        for r in results:
            fname = os.path.basename(r["path"])
            w.writerow([fname, r['pred'][0], r['pred'][1]])
    print(f"       CSV saved to {csv_path}")

    # 5. Visualize — individual images
    print("[4/5] Drawing individual results …")
    ind_dir = os.path.join(args.save_dir, "individual")
    os.makedirs(ind_dir, exist_ok=True)

    scale = _global_scale(results)
    print(f"       Global scale factor = {scale:.4f}")
    for r in results:
        fname = os.path.splitext(os.path.basename(r["path"]))[0]
        out = os.path.join(ind_dir, f"{fname}_pred.png")
        draw_prediction(r["path"], r["pred"], out, scale)

    # 6. Visualize — composite grid
    print(f"[5/5] Building composite grid ({args.grid[0]}x{args.grid[1]}) …")
    comp_path = os.path.join(args.save_dir, "composite.png")
    build_composite(results, comp_path, scale,
                    grid_size=tuple(args.grid))

    print(f"\nDone! Results saved to {args.save_dir}")
    print(f"  Individual: {ind_dir}/")
    print(f"  Composite:  {comp_path}")
    print(f"  CSV:        {csv_path}")


if __name__ == "__main__":
    main()
