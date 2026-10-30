"""判类器：最近质心（Nearest Centroid）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class NearestCentroid:
    """标准化 + 每类质心 + 最近欧氏距离。

    - 标准化用训练集的 mean/std，避免量纲大的特征（面积）压制量纲小的（填充率）。
    - 预测分数用 softmax(-dist/τ)，τ=1，只用于阈值扫描时的排序，不做概率校准。
    """

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.classes_: List[str] = []
        self.centroids_: Optional[np.ndarray] = None
        self.counts_: Dict[str, int] = {}

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "NearestCentroid":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(list(y))
        if X.ndim != 2:
            raise ValueError("X 必须是 (n_samples, n_features)")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        Z = (X - self.mean_) / self.std_
        self.classes_ = sorted(set(y.tolist()))
        self.centroids_ = np.stack([Z[y == c].mean(axis=0) for c in self.classes_])
        self.counts_ = {c: int((y == c).sum()) for c in self.classes_}
        return self

    def _dist(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.centroids_ is None:
            raise RuntimeError("先 fit")
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        d = Z[:, None, :] - self.centroids_[None, :, :]
        return np.sqrt((d * d).sum(axis=2))  # (n, n_classes)

    def predict(self, X: np.ndarray) -> List[str]:
        d = self._dist(X)
        return [self.classes_[i] for i in d.argmin(axis=1)]

    def predict_scores(self, X: np.ndarray) -> List[tuple[str, float]]:
        """返回 (类名, 置信分)。softmax(-d/τ) 只用于排序。"""
        d = self._dist(X)
        logits = -d / max(self.tau, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        idx = d.argmin(axis=1)
        return [(self.classes_[i], float(p[j, i])) for j, i in enumerate(idx)]
