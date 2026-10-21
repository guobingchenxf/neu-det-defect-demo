"""可视化：叠加框、中间结果拼图。只用 OpenCV，避免为了画几张图引入 matplotlib 依赖。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .common import Box
from .preprocess import TradParams, run_pipeline


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()


def draw_boxes(
    gray: np.ndarray,
    preds: Sequence[Box],
    gts: Sequence[Box] = (),
    path: Optional[str | Path] = None,
    scale: int = 2,
) -> np.ndarray:
    """GT 绿（实线）、预测红（虚线风格用细线表示）。放大后写盘，方便放进报告。"""
    img = to_bgr(gray)
    for b in gts:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 0), 2)
    for b in preds:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 1)
        cv2.putText(
            img,
            f"{b.cls[:3]}:{b.score:.2f}",
            (int(b.x1), max(9, int(b.y1) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
    return img


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    out = to_bgr(panel)
    cv2.putText(out, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def save_stages(gray: np.ndarray, params: TradParams, path: str | Path) -> np.ndarray:
    """把管线每一步贴成 2×3 拼图：原图 / 光照归一 / CLAHE / 梯度 / 二值掩膜 / 形态学后。

    这张图是报告里"传统方法为什么这么设计"的定性证据，也是面试时最容易被追问的地方。
    """
    st = run_pipeline(gray, params)
    panels = [
        _label(gray, "raw"),
        _label(st["corrected"], "flat-field"),
        _label(st["enhanced"], "CLAHE"),
        _label(st["grad"], "sobel |g|"),
        _label(st["mask"] * 255, "binary"),
        _label(st["refined"] * 255, "close+open+merge"),
    ]
    panels = [cv2.resize(p, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) for p in panels]
    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])
    grid = np.vstack([top, bottom])
    grid = cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)
    return grid


def overlay_mask(gray: np.ndarray, mask: np.ndarray, path: Optional[str | Path] = None) -> np.ndarray:
    base = to_bgr(gray)
    m = (mask > 0).astype(np.uint8)
    red = np.zeros_like(base)
    red[:, :, 2] = 255
    base = np.where(m[:, :, None] > 0, (0.5 * base + 0.5 * red).astype(np.uint8), base)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), base)
    return base


def pick_diverse(files: List[str], gts: Dict[str, List[Box]], n: int) -> List[str]:
    """按"每类先各取一张"的顺序挑图，避免 draw 出来的全是同一类。

    val 的前几张常常是 crazing（分层切分按主类排序），只看前 N 张会误判"所有图都这样"。
    """
    by_cls: Dict[str, List[str]] = {}
    for f in files:
        boxes = gts.get(f) or []
        key = boxes[0].cls if boxes else "background"
        by_cls.setdefault(key, []).append(f)
    out: List[str] = []
    depth = 0
    order = sorted(by_cls)
    while len(out) < min(n, len(files)) and depth < 50:
        for c in order:
            if depth < len(by_cls[c]):
                out.append(by_cls[c][depth])
                if len(out) >= n:
                    break
        depth += 1
    return out
