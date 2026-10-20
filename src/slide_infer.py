"""滑窗推理：patch 概率 → 热力图 → 连通域 → 框 → NMS。

关键工程点（也是这个 fallback 的"能不能上产线"的答案）：
- **步长决定成本**：patch/stride 组合直接决定每张图的 patch 数。stride=patch/2 时
  计算量是 stride=patch 的 4 倍，但框的定位精度好得多——报告里给的就是这条曲线。
- **连通域聚类**：热力图上相邻的高分 patch 必须合成一个框，否则一个缺陷报 10 个框，
  过杀率直接失控（这是"有模型能力但没后处理"的典型翻车）。
- **分数取区域均值**：取 max 会让单个噪声 patch 决定整块，取均值更稳；
  两者都可以，报告里说明选了哪个。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .cnn_model import load_model
from .common import (
    CLASSES,
    DEFECT,
    Box,
    evaluate_detections,
    load_annotations,
    load_splits,
    nms,
    resolve_dataset,
    save_json,
    save_preds,
    threshold_curve,
    set_seed,
)
from .patches import crop_patch, sliding_windows
from .preprocess import load_gray


@torch.no_grad()
def infer_image(
    model: torch.nn.Module,
    gray: np.ndarray,
    patch: int = 64,
    stride: int = 32,
    thr: float = 0.5,
    batch: int = 128,
) -> Tuple[List[Box], Dict[str, float], np.ndarray]:
    """返回 (boxes, 分段耗时ms, 分数热力图)。"""
    h, w = gray.shape[:2]
    t_pre = time.perf_counter()
    wins = sliding_windows(h, w, patch, stride)
    crops = []
    for x, y in wins:
        c = crop_patch(gray, x, y, patch)
        if c.shape[0] != patch or c.shape[1] != patch:
            c = cv2.resize(c, (patch, patch), interpolation=cv2.INTER_LINEAR)
        crops.append(c.astype(np.float32))
    patches = np.stack(crops)
    patches = (patches / 255.0 - 0.5) / 0.25
    x_t = torch.from_numpy(patches).unsqueeze(1)
    t_pre = time.perf_counter() - t_pre

    t_fwd = time.perf_counter()
    probs: List[np.ndarray] = []
    for i in range(0, len(x_t), batch):
        probs.append(torch.sigmoid(model(x_t[i : i + batch])).numpy())
    p = np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
    t_fwd = time.perf_counter() - t_fwd

    t_post = time.perf_counter()
    xs = sorted({x for x, _ in wins})
    ys = sorted({y for _, y in wins})
    nx, ny = len(xs), len(ys)
    grid = np.zeros((ny, nx), dtype=np.float32)
    xi_of = {x: i for i, x in enumerate(xs)}
    yi_of = {y: i for i, y in enumerate(ys)}
    for (x, y), s in zip(wins, p):
        grid[yi_of[y], xi_of[x]] = float(s)

    boxes = grid_to_boxes(grid, xs, ys, patch, thr)
    t_post = time.perf_counter() - t_post
    timings = {"preprocess_ms": t_pre * 1000, "forward_ms": t_fwd * 1000, "postprocess_ms": t_post * 1000}
    return boxes, timings, grid


def grid_to_boxes(
    grid: np.ndarray, xs: List[int], ys: List[int], patch: int, thr: float, nms_iou: float = 0.4
) -> List[Box]:
    """亮格子 → 框。

    关键：格子的坐标是**窗口左上角**，而模型判的是**窗口中心点**在不在缺陷里。
    所以框要由"亮格子的中心点"反推：一个格子代表它中心 ± patch/4 的一小块区域。
    早先直接拿窗口左上角 + patch 当框，框天生就大一圈（40 像素的缺陷出 96 像素的框），
    IoU 全部卡在 0.5 以下，mAP 只有 0.11 —— 这个 bug 不看可视化很难发现。
    """
    mask = (grid >= thr).astype(np.uint8)
    if mask.sum() == 0:
        return []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # 一个格子的覆盖半径应当等于半个步长（相邻格心之间的距离），而不是 patch/4：
    # 用 patch 尺度出框 → 框总是比缺陷大一圈；用固定小值 → 细步长时框又偏小。
    stride = int(xs[1] - xs[0]) if len(xs) > 1 else patch
    margin = max(2, stride // 2)
    h_img = max(ys[-1] + patch, patch) if ys else patch
    w_img = max(xs[-1] + patch, patch) if xs else patch
    raw: List[Box] = []
    for i in range(1, num):
        left, top, cw, ch, _area = (int(v) for v in stats[i])  # stats = [x, y, w, h, area]
        scores = grid[top : top + ch, left : left + cw]
        comp = labels[top : top + ch, left : left + cw] == i
        score = float(scores[comp].mean()) if comp.any() else 0.0
        cols = [xs[left + j] for j in range(cw) if comp[:, j].any()]
        rows = [ys[top + k] for k in range(ch) if comp[k, :].any()]
        if not cols or not rows:
            continue
        half = patch / 2.0
        x1 = float(np.clip(min(cols) + half - margin, 0, w_img))
        y1 = float(np.clip(min(rows) + half - margin, 0, h_img))
        x2 = float(np.clip(max(cols) + half + margin, 0, w_img))
        y2 = float(np.clip(max(rows) + half + margin, 0, h_img))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        raw.append(Box(DEFECT, x1, y1, x2, y2, score))
    return nms(raw, iou_thr=nms_iou)


def main() -> None:
    ap = argparse.ArgumentParser(description="滑窗 CNN 推理 + 评估")
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--model", default="results/cnn/best.pt")
    ap.add_argument("--which-split", dest="which_split", default="val", choices=["train", "tune", "val", "all"])
    ap.add_argument("--mode", default="eval", choices=["eval", "sweep", "draw"])
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--draw-n", type=int, default=6)
    ap.add_argument("--out-dir", default="results/cnn")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    set_seed(0)
    torch.set_num_threads(max(1, args.threads))
    model = load_model(args.model)
    img_dir, ann_dir = resolve_dataset(args.data_root)
    gts_all = load_annotations(img_dir, ann_dir)
    splits = load_splits(Path(args.splits))
    files = list(gts_all) if args.which_split == "all" else splits.get(args.which_split, [])
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"split={args.which_split} 为空")
    gts = {k: gts_all[k] for k in files}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer] {len(files)} 张，patch={args.patch} stride={args.stride} thr={args.thr} mode={args.mode}")
    t_all = time.perf_counter()
    preds: Dict[str, List[Box]] = {}
    timings: List[Dict[str, float]] = []
    grids: Dict[str, np.ndarray] = {}

    for name in files:
        gray = load_gray(img_dir / name)
        boxes, tm, grid = infer_image(model, gray, args.patch, args.stride, args.thr)
        preds[name] = boxes
        timings.append(tm)
        if args.mode in {"sweep", "draw"}:
            grids[name] = grid
        if args.mode == "draw":
            from .viz import draw_boxes

            draw_boxes(gray, boxes, gts[name], out_dir / "draw" / f"pred_{name}.png")
            heat = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
            heat = cv2.resize(heat, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.6, heat_color, 0.4, 0)
            cv2.imwrite(str(out_dir / "draw" / f"heat_{name}.png"), vis)

    total_s = time.perf_counter() - t_all
    ms_per_img = total_s / len(files) * 1000
    stage_avg = {k: round(float(np.mean([t[k] for t in timings])), 2) for k in timings[0]}
    print(f"[infer] 平均 {ms_per_img:.1f} ms/图（{stage_avg}）")

    m = evaluate_detections(preds, gts, iou_thr=args.iou, classes=(DEFECT,))
    m["ms_per_image"] = round(ms_per_img, 2)
    m["stage_ms"] = stage_avg
    m["patch"], m["stride"], m["thr"] = args.patch, args.stride, args.thr
    print(
        f"  mAP@0.5={m['mAP']} 召回={m['recall']} 漏检率={m['miss_rate']} "
        f"图级漏检={m['image_miss_rate']} 过杀={m['fp_per_image']}/图 框数={m['num_pred']}"
    )
    save_preds(
        out_dir / f"preds_{args.which_split}.json",
        preds,
        meta={"method": "sliding-cnn", "patch": args.patch, "stride": args.stride, "thr": args.thr, "ms_per_image": round(ms_per_img, 2)},
    )
    save_json(out_dir / f"metrics_{args.which_split}_p{args.patch}_s{args.stride}.json", m)

    if args.mode == "sweep":
        thrs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        curve = []
        for thr in thrs:
            p2 = {k: [b for b in v if b.score >= thr] for k, v in preds.items()}
            mm = evaluate_detections(p2, gts, iou_thr=args.iou, classes=(DEFECT,))
            curve.append({"thr": thr, "miss_rate": mm["miss_rate"], "fp_per_image": mm["fp_per_image"], "mAP": mm["mAP"]})
            print(f"  thr={thr}: 漏检率={mm['miss_rate']} 过杀={mm['fp_per_image']}/图 mAP={mm['mAP']}")
        save_json(out_dir / f"curve_p{args.patch}_s{args.stride}.json", curve)


if __name__ == "__main__":
    main()
