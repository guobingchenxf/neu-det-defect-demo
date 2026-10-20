"""传统图像处理基线（零训练）。

三个模式：
  tune      在 tune 集上网格搜索参数，落盘 best_params.json（**不碰 val**）
  eval      用给定参数在指定 split 上跑，输出预测 json + 指标 json + 阈值扫描
  classify  给传统框加一个最近质心判类器（6 类），用来暴露"定位可以、判类不行"
  draw      导几张叠加图做人工检查（报告里的定性证据）

用法示例：
  python -m src.trad_baseline --mode tune
  python -m src.trad_baseline --mode eval --split val
  python -m src.trad_baseline --mode classify --split val
  python -m src.trad_baseline --mode draw --split val --draw-n 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .classifier import NearestCentroid
from .common import (
    CLASSES,
    DEFECT,
    Box,
    evaluate_detections,
    list_images,
    load_annotations,
    load_splits,
    resolve_dataset,
    save_json,
    save_preds,
    threshold_curve,
    set_seed,
)
from .preprocess import TradParams, box_features, load_gray, predict_image, run_pipeline

GRID: Dict[str, List] = {
    "bin_method": ["otsu", "percentile"],
    "bin_scale": [0.8, 1.0, 1.2],
    "min_area": [30, 60, 120],
    "close_k": [3, 5],
    "merge_k": [1, 3],
}


# --------------------------------------------------------------------------- #
def precision_recall_f1(m: Dict[str, object]) -> Tuple[float, float, float]:
    rec = float(m.get("recall") or 0.0)
    n_gt = int(m.get("num_gt") or 0)
    n_pred = int(m.get("num_pred") or 0)
    tp = rec * n_gt
    prec = tp / n_pred if n_pred else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def run_split(
    files: Sequence[str],
    img_dir: Path,
    params: TradParams,
    limit: Optional[int] = None,
) -> Dict[str, List[Box]]:
    out: Dict[str, List[Box]] = {}
    for i, name in enumerate(files):
        if limit is not None and i >= limit:
            break
        gray = load_gray(img_dir / name)
        out[name] = predict_image(gray, params)
    return out


def iter_grid(grid: Dict[str, List]) -> Iterable[Dict[str, object]]:
    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


# --------------------------------------------------------------------------- #
def mode_tune(args: argparse.Namespace, img_dir: Path, gts: Dict[str, List[Box]], splits: Dict[str, List[str]]) -> TradParams:
    files = splits.get("tune") or splits.get("train")
    if not files:
        raise RuntimeError("tune 集为空，先跑 make_splits")
    if args.limit:
        files = files[: args.limit]
    gts_sub = {k: gts[k] for k in files}
    print(f"[tune] 在 {len(files)} 张图上搜索 {int(np.prod([len(v) for v in GRID.values()]))} 组参数")

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for ci, combo in enumerate(iter_grid(GRID), 1):
        p = TradParams.from_dict(combo)
        preds = run_split(files, img_dir, p)
        m = evaluate_detections(preds, gts_sub, iou_thr=args.iou, classes=(DEFECT,))
        prec, rec, f1 = precision_recall_f1(m)
        rows.append(
            {
                "combo": combo,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "fp_per_image": m["fp_per_image"],
                "image_miss_rate": m["image_miss_rate"],
                "elapsed": round(time.time() - t0, 1),
            }
        )
        print(
            f"  [{ci:>3}] scale={combo['bin_scale']} method={combo['bin_method']:<10} "
            f"min_area={combo['min_area']:>3} close={combo['close_k']} merge={combo['merge_k']} "
            f"→ P={prec:.3f} R={rec:.3f} F1={f1:.3f} FP/img={m['fp_per_image']:.2f}"
        )

    rows_sorted = sorted(rows, key=lambda r: -float(r["f1"]))
    best_f1 = rows_sorted[0]
    # 第二个交付点：在"高召回"约束下取过杀最低的一组（工业上常这么选）
    high_recall = [r for r in rows if float(r["recall"]) >= 0.85]
    best_hr = min(high_recall, key=lambda r: float(r["fp_per_image"])) if high_recall else None

    out_dir = Path(args.out_dir)
    save_json(out_dir / "grid_search.json", rows_sorted)
    best_params = TradParams.from_dict(best_f1["combo"])  # type: ignore[arg-type]
    save_json(
        out_dir / "best_params.json",
        {
            "objective": "max F1 @ IoU 0.5 on tune split",
            "params": best_params.to_dict(),
            "tune_metrics": {k: best_f1[k] for k in ("precision", "recall", "f1", "fp_per_image")},
            "high_recall_alternative": (
                {
                    "params": TradParams.from_dict(best_hr["combo"]).to_dict(),  # type: ignore[arg-type]
                    "tune_metrics": {k: best_hr[k] for k in ("precision", "recall", "f1", "fp_per_image")},
                }
                if best_hr
                else None
            ),
            "num_tune_images": len(files),
        },
    )
    print(f"\n[tune] best F1={best_f1['f1']:.3f} @ {best_f1['combo']}")
    if best_hr:
        print(f"[tune] 高召回备选 R={best_hr['recall']:.3f} FP/img={best_hr['fp_per_image']:.2f} @ {best_hr['combo']}")
    print(f"[ok] 参数写入 {out_dir/'best_params.json'}（注意：这些参数只在 tune 集上选过）")
    return best_params


# --------------------------------------------------------------------------- #
def load_params(args: argparse.Namespace) -> TradParams:
    if args.params:
        d = json.loads(Path(args.params).read_text(encoding="utf-8"))
        if "params" in d:
            d = d["params"]
        return TradParams.from_dict(d)
    p = Path(args.out_dir) / "best_params.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"[params] 使用 {p}（{d.get('objective', '')}）")
        return TradParams.from_dict(d["params"])
    print("[params] 没找到 best_params.json，使用默认参数（建议先 --mode tune）")
    return TradParams()


def mode_eval(
    args: argparse.Namespace,
    img_dir: Path,
    gts: Dict[str, List[Box]],
    splits: Dict[str, List[str]],
    params: TradParams,
) -> None:
    out_dir = Path(args.out_dir)
    targets = [args.split] if args.split != "all" else ["train", "tune", "val"]
    summary: Dict[str, object] = {"params": params.to_dict(), "iou_thr": args.iou, "splits": {}}

    for split in targets:
        files = splits.get(split) or []
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"[{split}] 空 split，跳过")
            continue
        print(f"\n[{split}] {len(files)} 张图")
        t0 = time.time()
        preds = run_split(files, img_dir, params)
        elapsed = time.time() - t0
        gts_sub = {k: gts[k] for k in files}
        m = evaluate_detections(preds, gts_sub, iou_thr=args.iou, classes=(DEFECT,))
        prec, rec, f1 = precision_recall_f1(m)
        m.update({"precision": round(prec, 4), "f1": round(f1, 4), "ms_per_image": round(elapsed / len(files) * 1000, 2)})
        print(
            f"  mAP@0.5={m['mAP']} R={m['recall']} 漏检率={m['miss_rate']} "
            f"图级漏检率={m['image_miss_rate']} 过杀={m['fp_per_image']} 框/图  F1={m['f1']} "
            f"{m['ms_per_image']}ms/图"
        )
        save_preds(
            out_dir / f"preds_{split}.json",
            preds,
            meta={
                "method": "trad",
                "split": split,
                "params": params.to_dict(),
                "ms_per_image": m["ms_per_image"],
            },
        )
        save_json(out_dir / f"metrics_{split}.json", m)
        summary["splits"][split] = m  # type: ignore[index]

        if args.sweep:
            thrs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
            curve = threshold_curve(preds, gts_sub, thrs, iou_thr=args.iou, classes=(DEFECT,))
            save_json(out_dir / f"curve_{split}.json", curve)
            print("  阈值扫描（score 是框内相对梯度强度）：")
            for c in curve:
                print(f"    thr={c['thr']:<5} 漏检率={c['miss_rate']} 过杀={c['fp_per_image']} 框/图")

    save_json(out_dir / "metrics_summary.json", summary)


# --------------------------------------------------------------------------- #
def collect_features(
    files: Sequence[str],
    img_dir: Path,
    gts: Dict[str, List[Box]],
    params: TradParams,
) -> Tuple[np.ndarray, List[str]]:
    """取 GT 框的手工特征做训练样本（不依赖检测结果，避免"用自己检出的框训自己"）。"""
    X: List[np.ndarray] = []
    y: List[str] = []
    for name in files:
        boxes = gts.get(name, [])
        if not boxes:
            continue
        gray = load_gray(img_dir / name)
        st = run_pipeline(gray, params)
        for b in boxes:
            if b.cls not in CLASSES:
                continue
            X.append(box_features(gray, st["enhanced"], st["grad"], b))
            y.append(b.cls)
    if not X:
        raise RuntimeError("没有可用的 GT 框，无法训练判类器")
    return np.stack(X), y


def mode_classify(
    args: argparse.Namespace,
    img_dir: Path,
    gts: Dict[str, List[Box]],
    splits: Dict[str, List[str]],
    params: TradParams,
) -> None:
    out_dir = Path(args.out_dir)
    train_files = splits.get("train") or []
    if args.limit:
        train_files = train_files[: args.limit]
    print(f"[classify] 用 {len(train_files)} 张图的 GT 框训练最近质心判类器")
    X, y = collect_features(train_files, img_dir, gts, params)
    clf = NearestCentroid().fit(X, y)
    print(f"[classify] 训练样本 {len(y)}，类分布 {clf.counts_}")
    save_json(out_dir / "classifier.json", {"classes": clf.classes_, "counts": clf.counts_, "template": "nearest-centroid"})

    split = args.split
    files = splits.get(split) or []
    if args.limit:
        files = files[: args.limit]
    preds: Dict[str, List[Box]] = {}
    for name in files:
        gray = load_gray(img_dir / name)
        st = run_pipeline(gray, params)
        from .preprocess import components_to_boxes

        boxes, _ = components_to_boxes(st["refined"], st["grad"], params)
        if boxes:
            F = np.stack([box_features(gray, st["enhanced"], st["grad"], b) for b in boxes])
            scored = clf.predict_scores(F)
            preds[name] = [
                Box(cls=c, x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, score=b.score * s)
                for b, (c, s) in zip(boxes, scored)
            ]
        else:
            preds[name] = []

    gts_sub = {k: gts[k] for k in files}
    m = evaluate_detections(preds, gts_sub, iou_thr=args.iou, classes=tuple(CLASSES))
    print("\n[classify] 直接给传统框贴类别（6 类，IoU 0.5）：")
    print(f"  mAP={m['mAP']} 类内召回={m['recall']} 过杀={m['fp_per_image']}")
    print("  逐类 AP：", {k: v for k, v in m["per_class_ap"].items()})  # type: ignore[union-attr]
    save_preds(out_dir / f"preds_classify_{split}.json", preds, meta={"method": "trad+nearest-centroid", "params": params.to_dict()})
    save_json(out_dir / f"metrics_classify_{split}.json", m)

    # 对照组：把同一个框全贴成同一个类，看"随便猜"能拿多少（AP 的随机基线）
    from collections import Counter

    maj = Counter(y).most_common(1)[0][0]
    preds_maj = {
        k: [Box(cls=maj, x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, score=b.score) for b in v] for k, v in preds.items()
    }
    m_maj = evaluate_detections(preds_maj, gts_sub, iou_thr=args.iou, classes=tuple(CLASSES))
    print(f"  对照（全部预测成多数类 {maj}）：mAP={m_maj['mAP']} 类内召回={m_maj['recall']}")
    save_json(out_dir / f"metrics_majority_{split}.json", m_maj)


# --------------------------------------------------------------------------- #
def mode_draw(
    args: argparse.Namespace,
    img_dir: Path,
    gts: Dict[str, List[Box]],
    splits: Dict[str, List[str]],
    params: TradParams,
) -> None:
    from .viz import draw_boxes, save_stages

    out_dir = Path(args.out_dir) / "draw"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = (splits.get(args.split) or [])[: args.draw_n]
    for name in files:
        gray = load_gray(img_dir / name)
        preds = predict_image(gray, params)
        draw_boxes(gray, preds, gts.get(name, []), out_dir / f"pred_{name}.png")
        save_stages(gray, params, out_dir / f"stages_{name}.png")
        print(f"  {name}: pred={len(preds)} gt={len(gts.get(name, []))}")
    print(f"[ok] 图写到 {out_dir}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="传统图像处理基线（零训练）")
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--out-dir", default="results/trad")
    ap.add_argument("--mode", choices=["tune", "eval", "classify", "draw"], default="eval")
    ap.add_argument("--split", default="val", choices=["train", "tune", "val", "all"])
    ap.add_argument("--params", default=None, help="指定参数 json（默认读 out-dir/best_params.json）")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（快速冒烟用，0=全部）")
    ap.add_argument("--sweep", action="store_true", help="eval 模式额外做阈值扫描")
    ap.add_argument("--draw-n", type=int, default=6)
    args = ap.parse_args()
    if args.limit == 0:
        args.limit = None

    set_seed(0)
    img_dir, ann_dir = resolve_dataset(args.data_root)
    gts = load_annotations(img_dir, ann_dir)
    splits = load_splits(Path(args.splits))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "tune":
        mode_tune(args, img_dir, gts, splits)
    elif args.mode == "eval":
        mode_eval(args, img_dir, gts, splits, load_params(args))
    elif args.mode == "classify":
        mode_classify(args, img_dir, gts, splits, load_params(args))
    else:
        mode_draw(args, img_dir, gts, splits, load_params(args))


if __name__ == "__main__":
    main()
