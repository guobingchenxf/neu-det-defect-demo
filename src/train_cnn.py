"""训练滑窗分类器（patch 级二分类）。

几个刻意的选择：
- **不用数据增强里的裁剪缩放**：滑窗在推理时会看到与原图同尺度的 patch，
  训练时把目标缩放会制造 train/test 尺度不一致。
- **pos_weight = neg/pos**：缺陷 patch 远少于背景 patch，不加权会退化成"全判背景"，
  表现为 AUC 还行但召回为 0——这个坑在报告里会作为一条失败记录。
- **val 用 patch 级 AUC 选模型**，但最终交付的指标一定是**框级**的（见 slide_infer），
  因为产线不关心 patch 准确率。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .cnn_model import build_model
from .common import (
    Box,
    load_annotations,
    load_splits,
    resolve_dataset,
    save_json,
    set_seed,
)
from .patches import PatchDataset, build_index
from .preprocess import load_gray


# --------------------------------------------------------------------------- #
def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC，带并列名次平均（不引入 sklearn 依赖）。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sorter = np.argsort(scores, kind="mergesort")
    sorted_s = scores[sorter]
    obs = np.r_[True, sorted_s[1:] != sorted_s[:-1]]
    inv = np.empty(len(scores), dtype=int)
    inv[sorter] = np.arange(len(scores))
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    ranks = 0.5 * (count[dense] + count[dense - 1] + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_split_images(
    files: List[str], img_dir: Path, gts: Dict[str, List[Box]], limit: int = 0
) -> Tuple[List[np.ndarray], List[List[Box]], List[str]]:
    if limit:
        files = files[:limit]
    images, boxes, names = [], [], []
    for name in files:
        images.append(load_gray(img_dir / name))
        boxes.append(gts.get(name, []))
        names.append(name)
    return images, boxes, names


@torch.no_grad()
def evaluate_patches(model: nn.Module, ds: PatchDataset, batch: int = 256) -> Dict[str, float]:
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x)).numpy())
        ys.append(y.view(-1).numpy())
    p = np.concatenate(ps)
    y = np.concatenate(ys).astype(np.int64)
    pred = (p >= 0.5).astype(np.int64)
    return {
        "auc": round(rank_auc(p, y), 4),
        "acc": round(float((pred == y).mean()), 4),
        "recall": round(float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan"), 4),
        "precision": round(
            float((y[pred == 1] == 1).mean()) if (pred == 1).any() else float("nan"), 4
        ),
        "num": int(len(y)),
        "pos_ratio": round(float(y.mean()), 4),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="训练滑窗 patch 分类器")
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out-dir", default="results/cnn")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--pos-per-img", type=int, default=6)
    ap.add_argument("--neg-per-img", type=int, default=10)
    ap.add_argument("--hard-neg-ratio", type=float, default=0.5)
    ap.add_argument("--max-train-images", type=int, default=0, help="只用前 N 张训练（冒烟用）")
    ap.add_argument("--max-val-images", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU 线程数；超订会反而变慢")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir, ann_dir = resolve_dataset(args.data_root)
    gts = load_annotations(img_dir, ann_dir)
    splits = load_splits(Path(args.splits))

    tr_imgs, tr_boxes, tr_names = load_split_images(splits.get("train", []), img_dir, gts, args.max_train_images)
    va_imgs, va_boxes, _ = load_split_images(splits.get("val", []), img_dir, gts, args.max_val_images)
    if not tr_imgs:
        raise SystemExit("train split 为空")
    print(f"[data] train {len(tr_imgs)} 张 / val {len(va_imgs)} 张（patch={args.patch}）")

    t0 = time.time()
    tr_idx = build_index(
        tr_imgs,
        tr_boxes,
        patch=args.patch,
        pos_per_img=args.pos_per_img,
        neg_per_img=args.neg_per_img,
        hard_neg_ratio=args.hard_neg_ratio,
        seed=args.seed,
    )
    va_idx = build_index(
        va_imgs,
        va_boxes,
        patch=args.patch,
        pos_per_img=args.pos_per_img,
        neg_per_img=args.neg_per_img,
        hard_neg_ratio=0.3,
        seed=args.seed + 1,
    )
    tr_ds = PatchDataset(tr_imgs, tr_idx, patch=args.patch, train=True, seed=args.seed)
    va_ds = PatchDataset(va_imgs, va_idx, patch=args.patch, train=False, seed=args.seed)
    print(f"[patch] train {tr_ds.stats()} / val {va_ds.stats()}，采样耗时 {time.time()-t0:.1f}s")

    model = build_model(width=args.width)
    print(f"[model] 参数量 {model.num_params()/1e6:.3f}M")
    pos = tr_ds.stats()["pos"]
    neg = tr_ds.stats()["neg"]
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)

    best = {"auc": -1.0, "epoch": -1}
    history: List[Dict[str, float]] = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        tot, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y.view(-1))
            loss.backward()
            opt.step()
            tot += float(loss.item()) * x.size(0)
            n += x.size(0)
        sched.step()
        vm = evaluate_patches(model, va_ds)
        # 监控用的"训练子集"必须同时裁 images 和 entries：
        # entries 里的 img_idx 指向 tr_imgs 的下标，只裁 images 会索引越界
        # （真实数据 500 张时立刻炸：IndexError: list index out of range）。
        sub_imgs = tr_imgs[:200]
        sub_idx = [e for e in tr_idx if e[0] < len(sub_imgs)][:2000]
        tm = evaluate_patches(model, PatchDataset(sub_imgs, sub_idx, patch=args.patch, train=False))
        row = {"epoch": ep, "loss": round(tot / max(n, 1), 4), **{f"val_{k}": v for k, v in vm.items()}, "train_auc": tm["auc"], "sec": round(time.time() - t_ep, 1)}
        history.append(row)  # type: ignore[arg-type]
        print(
            f"  ep{ep:>2} loss={row['loss']:.4f} val_auc={vm['auc']:.4f} val_recall={vm['recall']:.3f} "
            f"val_prec={vm['precision']:.3f} train_auc={tm['auc']:.4f} ({row['sec']}s)"
        )
        if vm["auc"] > best["auc"]:
            best = {"auc": vm["auc"], "epoch": ep}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {"width": args.width, "patch": args.patch, "dropout": 0.1},
                    "val": vm,
                },
                out_dir / "best.pt",
            )
    print(f"[train] best val_auc={best['auc']} @ epoch {best['epoch']} → {out_dir/'best.pt'}")
    save_json(
        out_dir / "train_log.json",
        {
            "args": vars(args),
            "history": history,
            "best": best,
            "patch_stats": {"train": tr_ds.stats(), "val": va_ds.stats()},
            "num_params": model.num_params(),
        },
    )
    (out_dir / "train_log.csv").write_text(
        "\n".join(
            ["epoch,loss,val_auc,val_acc,val_recall,val_precision,train_auc,sec"]
            + [
                f"{r['epoch']},{r['loss']},{r.get('val_auc')},{r.get('val_acc')},{r.get('val_recall')},{r.get('val_precision')},{r['train_auc']},{r['sec']}"
                for r in history
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
