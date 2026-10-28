"""Evaluate a trained classifier on the held-out test split and save plots.

Outputs (in --outdir): metrics.json, confusion_matrix.png, samples.png, errors.png

Usage:
    python eval_test.py --model MobileNetV2
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classification.predict import CLASS_NAMES, DrowsinessPredictor  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="MobileNetV2")
    p.add_argument("--data", default="data/processed/test")
    p.add_argument("--outdir", default="eval_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def grid(items, path, title, cols=6):
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (img_path, true, pred, conf) in zip(axes, items):
        ax.imshow(Image.open(img_path).convert("RGB"))
        ok = true == pred
        ax.set_title(f"T:{true} P:{pred}\n{conf:.2f}", fontsize=8,
                     color="green" if ok else "red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    data_dir = ROOT / args.data
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    predictor = DrowsinessPredictor(
        args.model, str(ROOT / "models" / "weights" / f"{args.model}_best.pt"))

    records = []  # (path, true, pred, conf of pred, p_drowsy)
    for cls in CLASS_NAMES:
        for f in sorted((data_dir / cls).iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            r = predictor.predict(Image.open(f).convert("RGB"))
            records.append((f, cls, r["class"], r["confidence"],
                            r["probabilities"]["drowsy"]))
    if not records:
        sys.exit(f"No images under {data_dir}")

    idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    cm = np.zeros((2, 2), dtype=int)  # rows = true, cols = pred
    for _, t, p, _, _ in records:
        cm[idx[t], idx[p]] += 1

    def prf(c):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        return pr, rc, f1

    acc = float(np.trace(cm) / cm.sum())
    metrics = {"model": args.model, "n": int(cm.sum()), "accuracy": acc,
               "confusion_matrix_rows_true_cols_pred": cm.tolist(),
               "classes": CLASS_NAMES}
    for c in CLASS_NAMES:
        pr, rc, f1 = prf(idx[c])
        metrics[c] = {"precision": pr, "recall": rc, "f1": f1}
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{args.model}  acc={acc:.3f}")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    sample = random.sample(records, min(24, len(records)))
    grid([(r[0], r[1], r[2], r[3]) for r in sample],
         outdir / "samples.png", "Random test samples (green=correct)")
    errors = sorted((r for r in records if r[1] != r[2]),
                    key=lambda r: -r[3])[:24]
    grid([(r[0], r[1], r[2], r[3]) for r in errors],
         outdir / "errors.png", "Most confident mistakes")

    print("=" * 50)
    print(f"model    : {args.model}   test images: {cm.sum()}")
    print(f"accuracy : {acc:.4f}")
    for c in CLASS_NAMES:
        m = metrics[c]
        print(f"{c:7s}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  f1 {m['f1']:.3f}")
    print(f"confusion (rows=true, cols=pred {CLASS_NAMES}):\n{cm}")
    print(f"saved    : {outdir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
