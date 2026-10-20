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
    fine: int = 0,
    expand: int = 16,
    fine_scale: float = 1.0,
    batch: int = 128,
) -> Tuple[List[Box], Dict[str, float], np.ndarray]:
    """返回 (boxes, 分段耗时ms, 分数热力图)。

    `fine > 0` 时启用两阶段：粗网格（stride）找候选 → 只在候选框内用 `fine` 步长细搜。
    这是滑窗路线唯一能把"32 像素格心分辨率"这个物理限制打掉的工程手段：
    全图细搜要 324 个窗口，两阶段只要 36 + 候选数×~100。
    """
    h, w = gray.shape[:2]
    t_pre = time.perf_counter()
    wins = sliding_windows(h, w, patch, stride)
    t_pre = time.perf_counter() - t_pre

    t_fwd = time.perf_counter()
    probs = forward_windows(model, gray, wins, patch, batch)
    t_fwd = time.perf_counter() - t_fwd

    t_post = time.perf_counter()
    grid = grid_from_windows(wins, probs)
    xs = sorted({x for x, _ in wins})
    ys = sorted({y for _, y in wins})
    boxes = grid_to_boxes(grid, xs, ys, patch, thr)

    n_fine = 0
    t_fine_s = 0.0
    if fine > 0 and boxes:
        t_fine = time.perf_counter()
        boxes, n_fine = refine_candidates(model, gray, boxes, patch, fine, thr, expand, batch, fine_scale)
        t_fine_s = time.perf_counter() - t_fine
        t_fwd += t_fine_s  # 细搜的前向算进 forward，口径统一
    t_post = time.perf_counter() - t_post - t_fine_s  # 细搜已计入 forward，别重复算进后处理

    timings = {
        "preprocess_ms": t_pre * 1000,
        "forward_ms": t_fwd * 1000,
        "postprocess_ms": t_post * 1000,
        "windows": float(len(wins) + n_fine),
    }
    return boxes, timings, grid


@torch.no_grad()
def forward_windows(
    model: torch.nn.Module, gray: np.ndarray, coords: List[Tuple[int, int]], patch: int, batch: int = 128
) -> np.ndarray:
    """对给定窗口坐标批量打分（crop + 归一化 + sigmoid）。"""
    crops = []
    for x, y in coords:
        c = crop_patch(gray, x, y, patch)
        if c.shape[0] != patch or c.shape[1] != patch:
            c = cv2.resize(c, (patch, patch), interpolation=cv2.INTER_LINEAR)
        crops.append(c.astype(np.float32))
    patches = np.stack(crops)
    patches = (patches / 255.0 - 0.5) / 0.25
    x_t = torch.from_numpy(patches).unsqueeze(1)
    out: List[np.ndarray] = []
    for i in range(0, len(x_t), batch):
        out.append(torch.sigmoid(model(x_t[i : i + batch])).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def grid_from_windows(wins: List[Tuple[int, int]], probs: np.ndarray) -> np.ndarray:
    xs = sorted({x for x, _ in wins})
    ys = sorted({y for _, y in wins})
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    xi_of = {x: i for i, x in enumerate(xs)}
    yi_of = {y: i for i, y in enumerate(ys)}
    for (x, y), s in zip(wins, probs):
        grid[yi_of[y], xi_of[x]] = float(s)
    return grid


def _axis_positions(start: int, end: int, patch: int, step: int, limit: int) -> List[int]:
    """在 [start, end) 内铺窗口左上角，保证覆盖到末端且不越界。"""
    start = max(0, min(start, limit - patch))
    end = max(start + patch, min(end, limit))
    pos = list(range(start, end - patch + 1, step)) or [start]
    if pos[-1] + patch < end:
        pos.append(end - patch)
    return sorted(set(pos))


@torch.no_grad()
def refine_candidates(
    model: torch.nn.Module,
    gray: np.ndarray,
    coarse: List[Box],
    patch: int,
    fine: int,
    thr: float,
    expand: int = 16,
    batch: int = 128,
    fine_scale: float = 1.0,
) -> Tuple[List[Box], int]:
    """对每个粗框做局部细搜：区域 = 粗框外扩 expand，窗口边长仍为 patch。

    细搜区域里可能同时有两三个相邻缺陷（粗网格把它们并成了一个框），
    所以这里**不假设一个候选只对应一个缺陷**：细网格上重新走一遍连通域，出几个框就收几个。
    """
    h, w = gray.shape[:2]
    out: List[Box] = []
    n_win = 0
    for b in coarse:
        xs = _axis_positions(int(round(b.x1)) - expand, int(round(b.x2)) + expand, patch, fine, w)
        ys = _axis_positions(int(round(b.y1)) - expand, int(round(b.y2)) + expand, patch, fine, h)
        coords = [(x, y) for y in ys for x in xs]
        probs = forward_windows(model, gray, coords, patch, batch)
        n_win += len(coords)
        grid = grid_from_windows(coords, probs)
        mask = grid >= thr
        if not mask.any():
            out.append(b)  # 细搜没响应 → 退回粗框，宁漏勿吞
            continue
        ii, jj = np.nonzero(mask)
        wts = grid[ii, jj].astype(np.float64)
        cxs = np.asarray(xs, dtype=np.float64)[jj] + patch / 2.0
        cys = np.asarray(ys, dtype=np.float64)[ii] + patch / 2.0
        # 软质心（分数加权）比"取亮格子的外接框"稳：单个边界格子闪烁不会把框拉走一格
        cx = float((wts * cxs).sum() / wts.sum())
        cy = float((wts * cys).sum() / wts.sum())
        # 细搜只看见缺陷"核心"的响应，而 GT 通常覆盖整个扩展区 → 需要 fine_scale 外扩；
        # 这个系数在 tune 集上选，val 只报数（否则就是拿验证集调参）
        half_x = max(patch / 4.0, float(cxs.max() - cxs.min()) / 2.0 + fine / 2.0) * fine_scale
        half_y = max(patch / 4.0, float(cys.max() - cys.min()) / 2.0 + fine / 2.0) * fine_scale
        x1 = float(np.clip(cx - half_x, 0, w))
        x2 = float(np.clip(cx + half_x, 0, w))
        y1 = float(np.clip(cy - half_y, 0, h))
        y2 = float(np.clip(cy + half_y, 0, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            out.append(b)
            continue
        out.append(Box(DEFECT, x1, y1, x2, y2, float(wts.max())))
    return nms(out, iou_thr=0.4), n_win
    return nms(out, iou_thr=0.4), n_win


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
    ap.add_argument("--fine-stride", dest="fine_stride", type=int, default=0,
                    help="两阶段的细搜步长（0=关闭）；建议 8，只在粗框内跑")
    ap.add_argument("--fine-expand", dest="fine_expand", type=int, default=16)
    ap.add_argument("--fine-scale", dest="fine_scale", type=float, default=1.0,
                    help="细搜框外扩倍数（在 tune 集上选，val 只报数）")
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

    print(
        f"[infer] {len(files)} 张，patch={args.patch} stride={args.stride} thr={args.thr} "
        f"fine={args.fine_stride or 'off'} mode={args.mode}"
    )
    t_all = time.perf_counter()
    preds: Dict[str, List[Box]] = {}
    timings: List[Dict[str, float]] = []
    grids: Dict[str, np.ndarray] = {}

    for name in files:
        gray = load_gray(img_dir / name)
        boxes, tm, grid = infer_image(
            model, gray, args.patch, args.stride, args.thr, args.fine_stride,
            args.fine_expand, args.fine_scale,
        )
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
    stage_avg["windows"] = int(np.mean([t["windows"] for t in timings]))
    print(f"[infer] 平均 {ms_per_img:.1f} ms/图（窗口数/图={stage_avg['windows']}，{stage_avg}）")

    m = evaluate_detections(preds, gts, iou_thr=args.iou, classes=(DEFECT,))
    m["ms_per_image"] = round(ms_per_img, 2)
    m["stage_ms"] = stage_avg
    m["patch"], m["stride"], m["thr"] = args.patch, args.stride, args.thr
    m["fine_stride"] = args.fine_stride
    print(
        f"  mAP@0.5={m['mAP']} 召回={m['recall']} 漏检率={m['miss_rate']} "
        f"图级漏检={m['image_miss_rate']} 过杀={m['fp_per_image']}/图 框数={m['num_pred']}"
    )
    save_preds(
        out_dir / f"preds_{args.which_split}.json",
        preds,
        meta={"method": "sliding-cnn", "patch": args.patch, "stride": args.stride,
              "fine_stride": args.fine_stride, "thr": args.thr, "ms_per_image": round(ms_per_img, 2)},
    )
    save_json(out_dir / f"metrics_{args.which_split}_p{args.patch}_s{args.stride}_f{args.fine_stride}.json", m)

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
