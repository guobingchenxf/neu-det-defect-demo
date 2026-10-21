"""把 ROI 六分类的混淆矩阵画成图（README 用）。

    python -m tools.plot_confusion --metrics results/cls/metrics.json --out docs/figs/05_cls_confusion.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/cls/metrics.json")
    ap.add_argument("--out", default="docs/figs/05_cls_confusion.png")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    rep = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    cm = np.asarray(rep["metrics"]["confusion"], dtype=np.float64)
    classes = rep["class_order"]
    short = [c.replace("_", "\n") for c in classes]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1, keepdims=True).max() if cm.size else 1)
    ax.set_xticks(range(len(classes)), short, fontsize=8, rotation=0)
    ax.set_yticks(range(len(classes)), short, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        f"ROI 6-class  acc={rep['metrics']['acc']:.3f}  macroF1={rep['metrics']['macro_f1']:.3f}",
        fontsize=10,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v:
                ax.text(
                    j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > cm[i, :].max() * 0.55 else "#333333",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
