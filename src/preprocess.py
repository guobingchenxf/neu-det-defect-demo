"""传统图像处理基线：可解释、零训练、参数可网格搜索。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .common import Box, DEFECT


@dataclass
class TradParams:
    """传统管线的全部可调参数。默认值是在 200×200 灰度图上调出来的起点，不是最优。"""

    # --- 预处理 ---
    bg_kernel: int = 31        # 背景校正核；0/1 表示关闭
    clip_limit: float = 2.0    # CLAHE 对比度上限
    tile: int = 8              # CLAHE 网格
    blur_k: int = 3            # 高斯核（奇数）
    # --- 二值化 ---
    bin_method: str = "otsu"   # otsu | percentile
    bin_scale: float = 1.0     # 阈值倍数：>1 更严格（漏检↑过杀↓）
    pct: float = 98.0          # percentile 模式的分位数
    dark_k: float = 0.0        # >0 时叠加"暗缺陷"通道，阈值 = 中位数 - dark_k·std
    # --- 形态学 ---
    close_k: int = 5
    open_k: int = 3
    merge_k: int = 3
    merge_iter: int = 1
    # --- 框筛选 ---
    min_area: int = 40
    max_area_ratio: float = 0.9  # 面积上限 = 该比例 × 图像面积
    min_fill: float = 0.12       # 面积 / 外接框面积
    min_aspect: float = 0.08     # min(w,h)/max(w,h)；细长划痕也要留下
    min_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TradParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]


def _odd(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 0
    return k if k % 2 == 1 else k + 1


# --------------------------------------------------------------------------- #
# 单步算子（拆开是为了可视化中间结果、也方便消融）
# --------------------------------------------------------------------------- #
def load_gray(path: str | Path) -> np.ndarray:
    """直接以灰度读入。工业相机多为单通道；彩色图转灰度会引入插值，这里不做无谓转换。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"读图失败: {path}")
    return img


def flat_field(gray: np.ndarray, kernel: int = 31) -> np.ndarray:
    """光照归一化：用大核中值估计背景，做除法。"""
    k = _odd(kernel)
    if k == 0:
        return gray.copy()
    bg = cv2.medianBlur(gray, k)
    out = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    if clip_limit <= 0:
        return gray.copy()
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    return op.apply(gray)


def blur(gray: np.ndarray, k: int = 3) -> np.ndarray:
    k = _odd(k)
    return gray.copy() if k == 0 else cv2.GaussianBlur(gray, (k, k), 0)


def grad_mag(gray: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值，缩放到 0~255 的 uint8 便于用 Otsu / 分位数定阈值。"""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mx = float(mag.max())
    if mx <= 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)
    return np.clip(mag / mx * 255.0, 0, 255).astype(np.uint8)


def threshold_map(grad8: np.ndarray, p: TradParams) -> Tuple[np.ndarray, float]:
    """返回 (mask, 实际阈值)。Otsu 与分位数两条路，后者是兜底。"""
    if p.bin_method == "otsu":
        thr, _ = cv2.threshold(grad8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(thr) * float(p.bin_scale)
    else:
        thr = float(np.percentile(grad8, float(p.pct))) * float(p.bin_scale)
    mask = (grad8.astype(np.float32) >= thr).astype(np.uint8)
    return mask, thr


def dark_mask(gray: np.ndarray, dark_k: float) -> np.ndarray:
    """暗缺陷通道：缺陷常比背景暗，用 median - k·std 抓低灰度区。"""
    if dark_k <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    med = float(np.median(gray))
    std = float(gray.std())
    thr = med - float(dark_k) * std
    return (gray.astype(np.float32) <= thr).astype(np.uint8)


def refine_mask(mask: np.ndarray, p: TradParams) -> np.ndarray:
    """闭→开→膨胀合并：先连断裂、再去噪点、最后把同缺陷碎块并起来。"""
    m = mask.copy()
    if p.close_k > 1:
        k = _odd(p.close_k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if p.open_k > 1:
        k = _odd(p.open_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if p.merge_k > 1:
        k = _odd(p.merge_k)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=int(p.merge_iter))
    return m


# --------------------------------------------------------------------------- #
# 主入口：单张图 → 框
# --------------------------------------------------------------------------- #
def run_pipeline(gray: np.ndarray, p: TradParams) -> Dict[str, np.ndarray]:
    """返回所有中间结果，方便可视化与调试（报告里会用到这几张图）。"""
    corrected = flat_field(gray, p.bg_kernel)
    enhanced = clahe(corrected, p.clip_limit, p.tile)
    smooth = blur(enhanced, p.blur_k)
    grad8 = grad_mag(smooth)
    m_grad, thr = threshold_map(grad8, p)
    m_dark = dark_mask(enhanced, p.dark_k)
    mask = np.clip(m_grad | m_dark, 0, 1).astype(np.uint8)
    refined = refine_mask(mask, p)
    return {
        "corrected": corrected,
        "enhanced": enhanced,
        "smooth": smooth,
        "grad": grad8,
        "mask": mask,
        "refined": refined,
        "thr": np.array([thr], dtype=np.float32),
    }


def components_to_boxes(
    refined: np.ndarray, mask: np.ndarray, grad8: np.ndarray, p: TradParams
) -> Tuple[List[Box], float]:
    """连通域 → 筛选 → 框。分数 = 框内平均梯度 / 梯度 p99.5（相对强度，便于阈值扫描）。

    `refined` 是做过闭/开/膨胀的"分组结果"，`mask` 是阈值化后的原始掩膜。
    膨胀核只负责把同一缺陷的碎片连成一个组；如果直接拿膨胀后的连通域算外接框，
    框会系统性偏大、IoU 被吃掉一大截（第一次跑 F1 只有 0.39 就是这个原因）。
    所以：**分组看 refined，出框看 refined ∩ mask**。
    这条值得在面试里主动讲——它说明"指标难看时先怀疑方法学，而不是先换模型"。
    """
    h, w = refined.shape[:2]
    total = float(h * w)
    num, labels, _, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    grad = grad8.astype(np.float32)
    ref99 = float(np.percentile(grad, 99.5)) or 1.0

    boxes: List[Box] = []
    for i in range(1, num):  # 0 是背景
        region = labels == i
        tight = region & (mask > 0)
        if int(tight.sum()) < 8:  # 碎片只剩几个像素时退回整组，避免出 1 像素框
            tight = region
        ys, xs = np.nonzero(tight)
        x1, y1 = float(xs.min()), float(ys.min())
        x2, y2 = float(xs.max() + 1), float(ys.max() + 1)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        area = float(len(xs))
        if area < p.min_area or area > p.max_area_ratio * total:
            continue
        fill = area / float(bw * bh)
        if fill < p.min_fill:
            continue
        if min(bw, bh) / max(bw, bh) < p.min_aspect:
            continue
        roi = grad[int(y1) : int(y2), int(x1) : int(x2)]
        score = float(roi.mean()) / max(ref99, 1e-6)
        if score < p.min_score:
            continue
        boxes.append(Box(DEFECT, x1, y1, x2, y2, score))
    return boxes, ref99


def predict_image(gray: np.ndarray, p: TradParams) -> List[Box]:
    """单张图推理：预处理 → 掩膜 → 框。"""
    stages = run_pipeline(gray, p)
    boxes, _ = components_to_boxes(stages["refined"], stages["mask"], stages["grad"], p)
    boxes.sort(key=lambda b: -b.score)
    return boxes


def predict_path(path: str | Path, p: TradParams) -> List[Box]:
    return predict_image(load_gray(path), p)


# --------------------------------------------------------------------------- #
# 特征（给"判类器"用；传统方法定位可以，判类基本靠手工特征，这正是它的短板）
# --------------------------------------------------------------------------- #
def box_features(gray: np.ndarray, enhanced: np.ndarray, grad8: np.ndarray, box: Box) -> np.ndarray:
    h, w = gray.shape[:2]
    x1, y1 = int(np.clip(box.x1, 0, w - 1)), int(np.clip(box.y1, 0, h - 1))
    x2, y2 = int(np.clip(box.x2, x1 + 2, w)), int(np.clip(box.y2, y1 + 2, h))

    roi_g = gray[y1:y2, x1:x2]
    roi_e = enhanced[y1:y2, x1:x2]
    roi_grad = grad8[y1:y2, x1:x2].astype(np.float32)

    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    area = float(box.area())
    fill = area / float(bw * bh)

    # 纹理：拉普拉斯方差（清晰度/细纹），梯度直方图的熵
    lap_var = float(cv2.Laplacian(roi_e, cv2.CV_32F).var())
    hist, _ = np.histogram(roi_grad, bins=16, range=(0, 255))
    prob = hist.astype(np.float64) / max(hist.sum(), 1)
    entropy = float(-(prob[prob > 0] * np.log2(prob[prob > 0])).sum())

    # 方向性：Sobel 方向的梯度能量集中度（细长划痕方向一致）
    gx = cv2.Sobel(roi_e.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_e.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    ang = np.arctan2(gy, gx)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    vx, vy = float((mag * np.cos(ang)).sum()), float((mag * np.sin(ang)).sum())
    orientation_strength = float(np.hypot(vx, vy) / mag.sum())  # 0=各向同性 1=单向

    # 形状：细长比、Hu 矩
    elong = max(bw, bh) / float(min(bw, bh))
    hu = cv2.HuMoments(cv2.moments((roi_grad > np.percentile(roi_grad, 80)).astype(np.uint8))).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    feats = np.concatenate(
        [
            np.array(
                [
                    np.log(area + 1.0),
                    np.log(bw + 1.0),
                    np.log(bh + 1.0),
                    elong,
                    fill,
                    float(roi_g.mean()) / 255.0,
                    float(roi_g.std()) / 255.0,
                    float(roi_e.mean()) / 255.0,
                    float(roi_e.std()) / 255.0,
                    float(roi_grad.mean()) / 255.0,
                    float(roi_grad.std()) / 255.0,
                    lap_var / 1000.0,
                    entropy,
                    orientation_strength,
                    box.score,
                ],
                dtype=np.float64,
            ),
            hu.astype(np.float64),
        ]
    )
    return feats
