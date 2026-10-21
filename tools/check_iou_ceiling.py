"""极端化检验：在"热力图完美"的前提下，这套滑窗的框能不能达到 IoU 0.5？

固定 patch + 固定步长 + 只由亮格子的中心点拼框，意味着可选框的尺寸只有
    side(k) = patch - 2*margin + k*stride   (k = 0,1,2,...)
这组尺寸之外，一个缺陷无论模型多准都框不到。这个脚本对每个 GT 穷举
可达尺寸/位置，报告"理论可达最大 IoU"，用来区分"模型不行"和"框的几何就不允许"。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import DEFECT, load_annotations, load_splits  # noqa: E402


def best_iou_for(b, patch, stride, margin, img=200):
    """遍历所有可达的 (k_w, k_h, 起点格子)，返回该 GT 能拿到的最大 IoU。"""
    base = patch - 2 * margin
    best = 0.0
    gx = list(range(0, img - patch + 1, stride))
    for kw in range(0, 8):
        w = base + kw * stride
        if w <= 0:
            continue
        for kh in range(0, 8):
            h = base + kh * stride
            if h <= 0:
                continue
            for x in gx:
                cx1 = x + patch / 2 - margin
                for y in gx:
                    cy1 = y + patch / 2 - margin
                    # 真实出框一定 clip 到图像内（slide_infer.grid_to_boxes 也是这么做的），
                    # 这里忘了 clip 会让大尺寸候选框伸出图像，union 虚大、IoU 假性偏低。
                    bx1 = max(0.0, min(float(img), cx1))
                    by1 = max(0.0, min(float(img), cy1))
                    bx2 = max(0.0, min(float(img), cx1 + w))
                    by2 = max(0.0, min(float(img), cy1 + h))
                    # 贴边延伸：slide_infer 里"组件含第一/最后一排窗口就延伸到图像边界"。
                    # 判据是**窗口**位置（x/y 是窗口左上角），不是框的位置。
                    if x == 0:
                        bx1 = 0.0
                    if y == 0:
                        by1 = 0.0
                    if x + kw * stride + patch >= img:
                        bx2 = float(img)
                    if y + kh * stride + patch >= img:
                        by2 = float(img)
                    if bx2 - bx1 < 2 or by2 - by1 < 2:
                        continue
                    ix1, iy1 = max(bx1, b.x1), max(by1, b.y1)
                    ix2, iy2 = min(bx2, b.x2), min(by2, b.y2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    if inter <= 0:
                        continue
                    union = w * h + b.area() - inter
                    if union > 0:
                        best = max(best, inter / union)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--patch", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.data_root)
    gts = load_annotations(root / "IMAGES", root / "ANNOTATIONS")
    names = load_splits(Path(args.splits))[args.split]

    for stride in (16, 32):
        margin = max(2, stride // 2)
        ious = []
        for n in names:
            for b in gts[n]:
                ious.append((best_iou_for(b, args.patch, stride, margin), n, b))
        vals = np.array([v for v, _, _ in ious])
        bad = [(v, n, b) for v, n, b in ious if v < 0.5]
        print("stride=%2d margin=%2d | GT 框 %d 个：可达 IoU 中位 %.3f，均值 %.3f，<0.5 的占 %.1f%%"
              % (stride, margin, len(vals), np.median(vals), vals.mean(), 100.0 * len(bad) / len(vals)))
        if bad[:5]:
            print("   达不到 0.5 的例子：",
                  ["%s %s %.0fx%.0f(上限%.2f)" % (n.replace('.jpg', ''), b.cls[:5], b.x2 - b.x1, b.y2 - b.y1, v)
                   for v, n, b in bad[:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
