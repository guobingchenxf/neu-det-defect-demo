"""滑窗 + 小 CNN 的数据侧：把检测问题降维成 patch 二分类。
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import Box

MIN_GT_COVER = 0.05   # 中心在缺陷外、但窗口仍覆盖 5% 以上缺陷 → 歧义样本，丢弃
# 负样本中心离任一 GT 的最小距离（单位：patch）。
# 合成数据上 0.5 够用；真实 NEU-DET 的框很大（pitted 常占满整图、crazing 中位 130px），
# 0.5 的环带里几乎找不到"干净背景"，250 张图才采到 749 个负样本、正负比 57:43，
# 于是 pos_weight=neg/pos<1 反向压制正类，推理时几乎没有格子能过 0.5 阈值（271 张只出 7 个框）。
NEG_MARGIN = 0.3


# --------------------------------------------------------------------------- #
# 几何工具
# --------------------------------------------------------------------------- #
def _patch_at(cx: int, cy: int, patch: int) -> Tuple[int, int, int, int]:
    x1 = int(round(cx - patch / 2))
    y1 = int(round(cy - patch / 2))
    return x1, y1, x1 + patch, y1 + patch


def _center_dist_to_boxes(cx: float, cy: float, boxes: Sequence[Box]) -> float:
    """中心点到最近 GT 框的欧氏距离（点在框内为 0）。"""
    if not boxes:
        return float("inf")
    best = float("inf")
    for b in boxes:
        dx = max(b.x1 - cx, 0.0, cx - b.x2)
        dy = max(b.y1 - cy, 0.0, cy - b.y2)
        best = min(best, float(np.hypot(dx, dy)))
    return best


def _iou_and_cover(x1: int, y1: int, x2: int, y2: int, boxes: Sequence[Box]) -> Tuple[float, float]:
    """返回 (最大 IoU, 窗口被缺陷覆盖的最大比例)。留着做歧义样本的判定与调试。"""
    best_iou, best_cover = 0.0, 0.0
    patch_area = float(max(1, (x2 - x1) * (y2 - y1)))
    for b in boxes:
        ix1, iy1 = max(x1, b.x1), max(y1, b.y1)
        ix2, iy2 = min(x2, b.x2), min(y2, b.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            continue
        union = patch_area + b.area() - inter
        if union > 0:
            best_iou = max(best_iou, inter / union)
        best_cover = max(best_cover, inter / patch_area)
    return best_iou, best_cover


# --------------------------------------------------------------------------- #
# 采样
# --------------------------------------------------------------------------- #
def crop_patch(img: np.ndarray, x: int, y: int, patch: int) -> np.ndarray:
    """越界用 reflect 补，避免边缘 patch 全黑（边缘缺陷在产线上很常见）。"""
    h, w = img.shape[:2]
    if x >= 0 and y >= 0 and x + patch <= w and y + patch <= h:
        return img[y : y + patch, x : x + patch]
    pad_l, pad_t = max(0, -x), max(0, -y)
    pad_r, pad_b = max(0, x + patch - w), max(0, y + patch - h)
    padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    return padded[max(0, y) : max(0, y) + patch, max(0, x) : max(0, x) + patch]


def build_index(
    images: Sequence[np.ndarray],
    gts_per_img: Sequence[Sequence[Box]],
    patch: int = 64,
    pos_per_img: int = 6,
    neg_per_img: int = 10,
    hard_neg_ratio: float = 0.5,
    seed: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """返回 [(img_idx, x, y, label)]，x/y 是 patch 左上角。

    hard_neg_ratio 的负样本贴着 GT 框外侧采（"差一点就是缺陷"），
    让模型学边界而不是只学背景纹理；纯随机负样本会把它教懒。
    """
    rng = random.Random(seed)
    entries: List[Tuple[int, int, int, int]] = []
    for idx, (img, boxes) in enumerate(zip(images, gts_per_img)):
        h, w = img.shape[:2]
        n_box = max(1, len(boxes))

        # 正样本：中心点落在 GT 内（等分给每个框，小缺陷也能被采到）
        if boxes:
            per_box = max(1, pos_per_img // n_box)
            for b in boxes:
                for _ in range(per_box):
                    cx = b.x1 + rng.uniform(0.15, 0.85) * (b.x2 - b.x1)
                    cy = b.y1 + rng.uniform(0.15, 0.85) * (b.y2 - b.y1)
                    x1, y1, _, _ = _patch_at(cx, cy, patch)
                    entries.append((idx, x1, y1, 1))

        def is_clean_negative(cx: float, cy: float) -> bool:
            if _center_dist_to_boxes(cx, cy, boxes) < patch * NEG_MARGIN:
                return False
            x1, y1, x2, y2 = _patch_at(cx, cy, patch)
            _, cover = _iou_and_cover(x1, y1, x2, y2, boxes)
            return cover <= MIN_GT_COVER

        n_hard = int(round(neg_per_img * hard_neg_ratio))
        # 硬负样本：贴着某个 GT 框外侧一圈
        if boxes:
            for _ in range(n_hard):
                for _try in range(60):
                    b = rng.choice(list(boxes))
                    ang = rng.uniform(0, 2 * np.pi)
                    dist = patch * rng.uniform(0.5, 0.85)
                    cx = (b.x1 + b.x2) / 2 + dist * np.cos(ang)
                    cy = (b.y1 + b.y2) / 2 + dist * np.sin(ang)
                    if is_clean_negative(cx, cy):
                        entries.append((idx, *_patch_at(cx, cy, patch)[:2], 0))
                        break
        # 随机负样本
        for _ in range(neg_per_img - n_hard):
            for _try in range(60):
                cx = rng.uniform(0, w)
                cy = rng.uniform(0, h)
                if is_clean_negative(cx, cy):
                    entries.append((idx, *_patch_at(cx, cy, patch)[:2], 0))
                    break
    rng.shuffle(entries)
    return entries


def augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """只用保形变换 + 光照抖动：工业图像里"亮度/对比度偏移"才是最真实的域偏移。"""
    p = patch.astype(np.float32)
    if rng.random() < 0.5:
        p = p[:, ::-1]
    if rng.random() < 0.5:
        p = p[::-1, :]
    if rng.random() < 0.3:  # 小角度旋转，用仿射实现，border 用 reflect
        ang = rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((p.shape[1] / 2, p.shape[0] / 2), ang, 1.0)
        p = cv2.warpAffine(p, M, (p.shape[1], p.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    alpha = 1.0 + rng.uniform(-0.15, 0.15)   # 对比度
    beta = rng.uniform(-12, 12)              # 亮度
    p = p * alpha + beta
    if rng.random() < 0.3:
        p = p + rng.normal(0, 3.0, size=p.shape)
    return np.clip(p, 0, 255).astype(np.uint8)


class PatchDataset(Dataset):
    """整图缓存在内存（200×200 灰度图，1800 张才 ~72MB），按需裁 patch。

    不预先裁好所有 patch 的原因：pos_per_img × neg_per_img × 图像数 会到几十万个样本，
    存成 float32 要几个 GB，而按需裁一张图只要几微秒。
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        entries: Sequence[Tuple[int, int, int, int]],
        patch: int = 64,
        train: bool = True,
        seed: int = 0,
    ):
        self.images = images
        self.entries = list(entries)
        self.patch = int(patch)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx, x, y, label = self.entries[i]
        p = crop_patch(self.images[img_idx], x, y, self.patch)
        if p.shape[0] != self.patch or p.shape[1] != self.patch:  # 极端越界兜底
            p = cv2.resize(p, (self.patch, self.patch), interpolation=cv2.INTER_LINEAR)
        if self.train:
            p = augment(p, self.rng)
        x_t = torch.from_numpy(np.ascontiguousarray(p)).float().div_(255.0).sub_(0.5).div_(0.25).unsqueeze(0)
        return x_t, torch.tensor([float(label)], dtype=torch.float32)

    def stats(self) -> dict:
        pos = sum(1 for e in self.entries if e[3] == 1)
        return {"total": len(self.entries), "pos": pos, "neg": len(self.entries) - pos}


def sliding_windows(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """返回所有 patch 左上角坐标。保证右/下边界被覆盖（最后一个窗口贴边）。"""
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    ys = list(range(0, max(h - patch, 0) + 1, stride))
    if xs and xs[-1] != max(w - patch, 0):
        xs.append(max(w - patch, 0))
    if ys and ys[-1] != max(h - patch, 0):
        ys.append(max(h - patch, 0))
    return [(x, y) for y in ys for x in xs]
