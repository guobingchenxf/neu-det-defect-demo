"""临时调试：看单张图的滑窗热力图与出框，定位"patch AUC 0.98 但检测 mAP 0.11"的原因。"""
import sys
from pathlib import Path

import numpy as np
import torch

from src.common import load_annotations, load_splits, resolve_dataset
from src.preprocess import load_gray
from src.slide_infer import infer_image
from src.cnn_model import load_model

torch.set_num_threads(4)
img_dir, ann_dir = resolve_dataset("data/FAKE-NEU-DET")
gts = load_annotations(img_dir, ann_dir)
splits = load_splits(Path("data/splits.json"))
model = load_model("results/cnn/best.pt")

for name in splits["val"][:6]:
    gray = load_gray(img_dir / name)
    boxes, timings, grid = infer_image(model, gray, patch=64, stride=32, thr=0.5)
    print(f"\n### {name}  grid {grid.shape}  max={grid.max():.3f} mean={grid.mean():.3f}")
    print("GT:", [(round(b.x1), round(b.y1), round(b.x2), round(b.y2)) for b in gts[name]])
    print("PR:", [(round(b.x1), round(b.y1), round(b.x2), round(b.y2), round(b.score, 2)) for b in boxes])
    print("热力图行优先（>0.5 记 #，>0.2 记 +）:")
    for row in grid:
        print("   " + "".join("#" if v >= 0.5 else ("+" if v >= 0.2 else ".") for v in row))
