"""ROI 六分类：工业质检真正好用的那半步。

为什么另开一个任务而不是继续修滑窗检测：
- 定位交给传统方法/连通域，CNN 只回答"这一小块属于哪类缺陷"，是产线常见分工
  （对应 NIII 的"小模型初筛 + 大模型复判"里的小模型那一层）；
- NEU-DET 的框大量贴边、尺度跨 14~200px，固定窗口分类器在定位上先天吃亏，
  但在"给定 ROI 判类别"上，CNN 的语义能力是真能兑现的；
- 这个任务能在 CPU 上 2~3 分钟训完，适合放进 demo 让人一键复现。

用法：
    python -m src.train_cls --data-root data/NEU-DET --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .cnn_model import build_model
from .common import CLASSES, load_annotations, load_splits, save_json, set_seed
from .preprocess import load_gray

CTX = 0.2  # ROI 在 GT 框外扩 20%，把缺陷周边的纹理上下文也带进来


# --------------------------------------------------------------------------- #
def roi_of(img: np.ndarray, box, ctx: float = CTX) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1 = int(max(0, box.x1 - ctx * bw))
    y1 = int(max(0, box.y1 - ctx * bh))
    x2 = int(min(w, box.x2 + ctx * bw))
    y2 = int(min(h, box.y2 + ctx * bh))
    if x2 - x1 < 4 or y2 - y1 < 4:
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    return img[y1:y2, x1:x2]


def to_tensor(roi: np.ndarray, size: int) -> np.ndarray:
    r = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    return ((r.astype(np.float32) / 255.0) - 0.5) / 0.25


def augment(roi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        roi = roi[:, ::-1]
    if rng.random() < 0.5:
        roi = roi[::-1, :]
    if rng.random() < 0.6:
        h, w = roi.shape
        ang = rng.uniform(-8, 8)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        roi = cv2.warpAffine(roi, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.6:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-12, 12)
        roi = np.clip(roi.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        noise = rng.normal(0, 3.0, roi.shape)
        roi = np.clip(roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(roi)


class RoiDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, int]], size: int, train: bool, seed: int = 0):
        self.items = items
        self.size = size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        roi, y = self.items[i]
        if self.train:
            roi = augment(roi, self.rng)
        x = to_tensor(roi, self.size)
        return torch.from_numpy(x[None, :, :]), torch.tensor(y, dtype=torch.long)


# --------------------------------------------------------------------------- #
def collect(items_by_name: Dict[str, List], names: List[str], img_dir: Path):
    items: List[Tuple[np.ndarray, int]] = []
    cache: Dict[str, np.ndarray] = {}
    for n in names:
        for b in items_by_name.get(n, []):
            if b.cls not in CLASSES:
                continue
            if n not in cache:
                cache[n] = load_gray(img_dir / n)
            items.append((roi_of(cache[n], b), CLASSES.index(b.cls)))
    return items


def confusion_matrix(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    for yi, pi in zip(y, p):
        m[yi, pi] += 1
    return m


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, ds: RoiDataset, batch: int = 128):
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(model(x).argmax(1).numpy())
        ys.append(y.numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)
    cm = confusion_matrix(y, p, len(CLASSES))
    return {
        "acc": round(float((p == y).mean()), 4),
        "macro_f1": round(macro_f1(cm), 4),
        "per_class_recall": {
            CLASSES[i]: round(float(cm[i, i] / max(1, cm[i].sum())), 4) for i in range(len(CLASSES))
        },
        "num": int(len(y)),
        "confusion": cm.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="NEU-DET ROI 六分类（GT 框作为定位）")
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out-dir", default="results/cls")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.data_root)
    img_dir = root / "IMAGES"
    gts = load_annotations(img_dir, root / "ANNOTATIONS")
    splits = load_splits(Path(args.splits))

    t0 = time.time()
    tr = collect(gts, splits["train"], img_dir)
    va = collect(gts, splits["val"], img_dir)
    print(f"[data] train ROI {len(tr)} / val ROI {len(va)}（GT 框当定位，分类上限；裁图 {time.time()-t0:.1f}s）")
    if not tr or not va:
        raise SystemExit("样本为空")

    tr_ds = RoiDataset(tr, args.size, train=True, seed=args.seed)
    va_ds = RoiDataset(va, args.size, train=False)

    model = build_model(width=args.width, num_classes=len(CLASSES))
    print(f"[model] 参数量 {model.num_params()/1e6:.3f}M，输入 {args.size}x{args.size}")

    counts = np.bincount([y for _, y in tr], minlength=len(CLASSES)).astype(np.float64)
    weights = torch.tensor((counts.sum() / (len(CLASSES) * np.maximum(counts, 1))), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    hist = []
    best = {"acc": -1.0, "epoch": -1}
    for ep in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        tot, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.size(0)
            n += x.size(0)
        sched.step()
        vm = evaluate(model, va_ds)
        hist.append({"epoch": ep, "loss": round(tot / max(n, 1), 4), "val_acc": vm["acc"],
                     "val_f1": vm["macro_f1"], "sec": round(time.time() - t_ep, 1)})
        print(f"  ep{ep:>2} loss={hist[-1]['loss']:.4f} val_acc={vm['acc']:.4f} macroF1={vm['macro_f1']:.4f} ({hist[-1]['sec']}s)")
        if vm["acc"] > best["acc"]:
            best = {"acc": vm["acc"], "epoch": ep}
            torch.save({"state_dict": model.state_dict(),
                        "config": {"width": args.width, "num_classes": len(CLASSES),
                                   "patch": args.size, "dropout": 0.1},
                        "classes": CLASSES, "val": vm}, out_dir / "best.pt")

    # 单张 ROI 的推理延迟（P50/P95），ROI 分类是要跟定位拼在一起的，延迟必须单独报
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu")["state_dict"])
    model.eval()
    sample = torch.from_numpy(to_tensor(va[0][0], args.size)[None, None, :, :])
    lats = []
    with torch.no_grad():
        for _ in range(50):
            t = time.perf_counter()
            model(sample)
            lats.append((time.perf_counter() - t) * 1000)
    lats.sort()

    report = {
        "args": vars(args),
        "history": hist,
        "best": best,
        "metrics": evaluate(model, va_ds),
        "latency_ms": {"p50": round(float(np.percentile(lats, 50)), 3),
                       "p95": round(float(np.percentile(lats, 95)), 3)},
        "num_params": model.num_params(),
        "class_order": CLASSES,
    }
    save_json(out_dir / "metrics.json", report)
    cm = report["metrics"]["confusion"]
    lines = ["true\\pred," + ",".join(CLASSES)]
    for i, c in enumerate(CLASSES):
        lines.append(c + "," + ",".join(str(v) for v in cm[i]))
    (out_dir / "confusion.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[result] best val_acc={best['acc']:.4f} @ep{best['epoch']} macroF1={report['metrics']['macro_f1']:.4f}")
    print("[result] 逐类召回:", report["metrics"]["per_class_recall"])
    print(f"[result] ROI 推理 P50={report['latency_ms']['p50']}ms P95={report['latency_ms']['p95']}ms")
    print(f"[ok] {out_dir/'metrics.json'}  {out_dir/'confusion.csv'}")


if __name__ == "__main__":
    main()
