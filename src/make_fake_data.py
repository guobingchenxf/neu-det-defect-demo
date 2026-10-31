"""生成合成"假 NEU-DET"，用于在没有真实数据时把整条管线跑通。
六类按 NEU-DET 的语义：
  crazing          细密低对比度裂纹网络（传统方法的重灾区）
  inclusion        条状深色夹杂
  patches          大面积不规则斑块
  pitted_surface   密集小麻点（小目标）
  rolled-in_scale  不规则氧化皮压入（边界抖）
  scratches        方向一致的细划痕
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .common import CLASSES, Box


# --------------------------------------------------------------------------- #
def make_background(rng: np.random.Generator, size: int = 200) -> np.ndarray:
    """带低频光照不均 + 轧制方向纹理 + 噪声的灰度背景。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = rng.uniform(120, 175)
    # 低频不均匀（真实带钢一定有）
    gx, gy = rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)
    low = base + gx * (xx - size / 2) + gy * (yy - size / 2)
    # 轧制方向的细条纹
    streak = np.zeros((size, size), np.float32)
    for _ in range(rng.integers(1, 4)):
        f = rng.uniform(0.05, 0.12)
        ph = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(1.5, 5.0)
        streak += (amp * np.sin(f * xx + ph)).astype(np.float32)
    noise = rng.normal(0, rng.uniform(2.0, 5.0), size=(size, size)).astype(np.float32)
    img = low + streak + noise
    return np.clip(img, 0, 255)


def _clip_box(x1: float, y1: float, x2: float, y2: float, size: int, pad: int = 2) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(x1, 0, size - 2))
    y1 = int(np.clip(y1, 0, size - 2))
    x2 = int(np.clip(x2, x1 + pad, size))
    y2 = int(np.clip(y2, y1 + pad, size))
    return x1, y1, x2, y2


def add_scratches(img: np.ndarray, rng: np.random.Generator) -> List[Box]:
    size = img.shape[0]
    boxes: List[Box] = []
    for _ in range(rng.integers(1, 4)):
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(40, 120)
        cx, cy = rng.uniform(30, size - 30), rng.uniform(30, size - 30)
        dx, dy = np.cos(ang) * length / 2, np.sin(ang) * length / 2
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        val = rng.uniform(60, 110)  # 比背景暗
        cv2.line(img, p1, p2, float(val), thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
        xs, ys = [p1[0], p2[0]], [p1[1], p2[1]]
        boxes.append(Box("scratches", *_clip_box(min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2, size)))
    return boxes


def add_crazing(img: np.ndarray, rng: np.random.Generator) -> List[Box]:
    """细密低对比度裂纹网：传统梯度阈值最难的一类。"""
    size = img.shape[0]
    cx, cy = rng.uniform(70, size - 70), rng.uniform(70, size - 70)
    r = rng.uniform(28, 55)
    n = int(rng.integers(8, 18))
    pts = []
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        rr = rng.uniform(0.2, 1.0) * r
        p1 = (int(cx + rr * np.cos(a)), int(cy + rr * np.sin(a)))
        a2 = a + rng.uniform(-0.7, 0.7)
        rr2 = rr + rng.uniform(6, 20)
        p2 = (int(cx + rr2 * np.cos(a2)), int(cy + rr2 * np.sin(a2)))
        delta = rng.uniform(8, 18)  # 低对比度是这类缺陷的本质
        base = float(img[int(cy), int(cx)])
        cv2.line(img, p1, p2, float(np.clip(base - delta, 0, 255)), 1, cv2.LINE_AA)
        pts += [p1, p2]
        # 分叉
        if rng.random() < 0.5:
            p3 = (int(p2[0] + rng.uniform(-14, 14)), int(p2[1] + rng.uniform(-14, 14)))
            cv2.line(img, p2, p3, float(np.clip(base - delta, 0, 255)), 1, cv2.LINE_AA)
            pts.append(p3)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [Box("crazing", *_clip_box(min(xs), min(ys), max(xs), max(ys), size))]


def add_inclusion(img: np.ndarray, rng: np.random.Generator) -> List[Box]:
    size = img.shape[0]
    boxes: List[Box] = []
    for _ in range(rng.integers(1, 3)):
        cx, cy = rng.uniform(40, size - 40), rng.uniform(40, size - 40)
        a, b = rng.uniform(14, 34), rng.uniform(4, 9)
        ang = rng.uniform(0, 180)
        delta = rng.uniform(25, 55)
        base = float(img[int(cy), int(cx)])
        cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, float(np.clip(base - delta, 0, 255)), -1, cv2.LINE_AA)
        boxes.append(Box("inclusion", *_clip_box(cx - a - 3, cy - a - 3, cx + a + 3, cy + a + 3, size)))
    return boxes


def add_patches(img: np.ndarray, rng: np.random.Generator) -> List[Box]:
    size = img.shape[0]
    cx, cy = rng.uniform(60, size - 60), rng.uniform(60, size - 60)
    rx, ry = rng.uniform(20, 45), rng.uniform(15, 40)
    n = int(rng.integers(8, 14))
    angs = np.sort(rng.uniform(0, 2 * np.pi, n))
    rads = rng.uniform(0.65, 1.15, n)
    poly = np.stack(
        [cx + rx * rads * np.cos(angs), cy + ry * rads * np.sin(angs)], axis=1
    ).astype(np.int32)
    delta = rng.uniform(15, 30) * rng.choice([-1, 1])
    base = float(img[int(cy), int(cx)])
    cv2.fillPoly(img, [poly], float(np.clip(base + delta, 0, 255)), cv2.LINE_AA)
    xs, ys = poly[:, 0], poly[:, 1]
    return [Box("patches", *_clip_box(xs.min(), ys.min(), xs.max(), ys.max(), size))]


def add_pitted(img: np.ndarray, rng: np.random.Generator) -> List[Box]:
    size = img.shape[0]
    cx, cy = rng.uniform(60, size - 60), rng.uniform(60, size - 60)
    spread = rng.uniform(25, 45)
    n = int(rng.integers(15, 45))
    xs, ys = [], []
    for _ in range(n):
        px = int(np.clip(cx + rng.normal(0, spread), 2, size - 3))
        py = int(np.clip(cy + rng.normal(0, spread), 2, size - 3))
        r = int(rng.integers(1, 4))
        base = float(img[py, px])
        cv2.circle(img, (px, py), r, float(np.clip(base - rng.uniform(25, 60), 0, 255)), -1, cv2.LINE_AA)
        xs.append(px)
        ys.append(py)
    return [Box("pitted_surface", *_clip_box(min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3, size))]


def add_rolled_in_scale(img: np.ndarray, rng: np.random.Generator) -> List[Box]:
    size = img.shape[0]
    boxes: List[Box] = []
    for _ in range(rng.integers(1, 3)):
        cx, cy = rng.uniform(50, size - 50), rng.uniform(50, size - 50)
        n = int(rng.integers(6, 11))
        angs = np.sort(rng.uniform(0, 2 * np.pi, n))
        r0 = rng.uniform(16, 32)
        rads = r0 * rng.uniform(0.6, 1.4, n)
        poly = np.stack([cx + rads * np.cos(angs), cy + rads * np.sin(angs)], axis=1).astype(np.int32)
        base = float(img[int(cy), int(cx)])
        cv2.fillPoly(img, [poly], float(np.clip(base - rng.uniform(30, 60), 0, 255)), cv2.LINE_AA)
        # 压入物边缘有碎屑
        for _ in range(rng.integers(3, 8)):
            px = int(np.clip(cx + rng.normal(0, r0), 0, size - 1))
            py = int(np.clip(cy + rng.normal(0, r0), 0, size - 1))
            cv2.circle(img, (px, py), int(rng.integers(1, 3)), float(np.clip(base - rng.uniform(20, 50), 0, 255)), -1, cv2.LINE_AA)
        xs, ys = poly[:, 0], poly[:, 1]
        boxes.append(Box("rolled-in_scale", *_clip_box(xs.min(), ys.min(), xs.max(), ys.max(), size)))
    return boxes


PAINTERS = {
    "crazing": add_crazing,
    "inclusion": add_inclusion,
    "patches": add_patches,
    "pitted_surface": add_pitted,
    "rolled-in_scale": add_rolled_in_scale,
    "scratches": add_scratches,
}


# --------------------------------------------------------------------------- #
def write_voc(path: Path, filename: str, size: int, boxes: Sequence[Box]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "IMAGES"
    ET.SubElement(root, "filename").text = filename
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text = "NEU-DET-SYNTHETIC"
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text = str(size)
    ET.SubElement(sz, "height").text = str(size)
    ET.SubElement(sz, "depth").text = "1"
    ET.SubElement(root, "segmented").text = "0"
    for b in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = b.cls
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        # VOC 是 1-based 且 xmax/ymax 含端点：内部半开区间 [x1, x2) → xmin=x1+1, xmax=x2
        ET.SubElement(bb, "xmin").text = str(int(b.x1) + 1)
        ET.SubElement(bb, "ymin").text = str(int(b.y1) + 1)
        ET.SubElement(bb, "xmax").text = str(int(b.x2))
        ET.SubElement(bb, "ymax").text = str(int(b.y2))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成 NEU-DET 数据（用于跑通管线）")
    ap.add_argument("--out", default="data/FAKE-NEU-DET")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    (out / "IMAGES").mkdir(parents=True, exist_ok=True)
    (out / "ANNOTATIONS").mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for cls in CLASSES:
        painter = PAINTERS[cls]
        for i in range(args.per_class):
            img = make_background(rng, args.size)
            boxes = painter(img, rng)
            # 轻微失焦 + 压缩噪声，模拟真实相机链路（这一步会让纯阈值方法更难）
            img8 = np.clip(img, 0, 255).astype(np.uint8)
            img8 = cv2.GaussianBlur(img8, (3, 3), 0.6)
            name = f"{cls}_{i:03d}.jpg"
            cv2.imwrite(str(out / "IMAGES" / name), img8, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            write_voc(out / "ANNOTATIONS" / f"{cls}_{i:03d}.xml", name, args.size, boxes)
            total_boxes += len(boxes)
        print(f"  {cls:<16} {args.per_class} 张")
    print(f"[ok] 合成数据写到 {out}（{len(CLASSES)*args.per_class} 张图，{total_boxes} 个框）")
    print("     下一步：python -m src.make_splits --data-root " + str(out) + " --out data/splits.json")


if __name__ == "__main__":
    main()
