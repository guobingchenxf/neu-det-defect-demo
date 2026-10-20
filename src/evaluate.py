"""统一评估入口：把任意一份/多份预测 json 放在同一把尺子上量。

有了它，"传统基线 vs 滑窗 CNN"的主结果表就是一条命令生成的，
不是手工抄数字——这也是报告可复现的一部分。

用法：
  python -m src.evaluate --preds results/trad/preds_val.json --split val
  python -m src.evaluate \
      --preds results/trad/preds_val.json results/cnn/preds_val.json \
      --labels 传统基线 滑窗CNN --split val --sweep --out results/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import (
    CLASSES,
    DEFECT,
    Box,
    evaluate_detections,
    load_annotations,
    load_preds,
    load_splits,
    resolve_dataset,
    save_json,
    threshold_curve,
)


def metrics_row(m: Dict[str, object]) -> Dict[str, object]:
    rec = float(m.get("recall") or 0.0)
    n_gt, n_pred = int(m.get("num_gt") or 0), int(m.get("num_pred") or 0)
    tp = rec * n_gt
    prec = tp / n_pred if n_pred else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "f1": round(f1, 4), **m}


def md_table(rows: Sequence[Dict[str, object]]) -> str:
    head = ["方法", "mAP@0.5", "最差类AP", "召回", "漏检率", "图级漏检", "过杀/图", "P(ms)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        per = r.get("per_class_ap") or {}
        vals = [v for v in per.values() if v is not None] if isinstance(per, dict) else []
        worst = round(min(vals), 4) if vals else "—"
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r.get("label", "?"),
                r.get("mAP"),
                worst,
                r.get("recall"),
                r.get("miss_rate"),
                r.get("image_miss_rate"),
                r.get("fp_per_image"),
                r.get("ms_per_image", "—"),
            )
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="统一评估：多份预测 → 主结果表")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="val", choices=["train", "tune", "val", "all"])
    ap.add_argument("--classes", default="defect", choices=["defect", "all"], help="defect=类别无关定位；all=6 类")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    img_dir, ann_dir = resolve_dataset(args.data_root)
    gts_all = load_annotations(img_dir, ann_dir)
    splits = load_splits(Path(args.splits))
    files = list(gts_all) if args.split == "all" else splits.get(args.split, [])
    if not files:
        raise SystemExit(f"split={args.split} 为空")
    gts = {k: gts_all[k] for k in files}
    classes: Sequence[str] = (DEFECT,) if args.classes == "defect" else tuple(CLASSES)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or [Path(p).stem for p in args.preds]
    if len(labels) != len(args.preds):
        raise SystemExit("--labels 数量要和 --preds 一致")

    rows: List[Dict[str, object]] = []
    detail: Dict[str, object] = {"split": args.split, "iou_thr": args.iou, "classes": list(classes), "methods": {}}
    for label, path in zip(labels, args.preds):
        preds_all, meta = load_preds(Path(path))
        preds = {k: preds_all.get(k, []) for k in files}
        m = metrics_row(evaluate_detections(preds, gts, iou_thr=args.iou, classes=classes))
        m["label"] = label
        m["preds_path"] = str(path)
        m["ms_per_image"] = meta.get("ms_per_image") if isinstance(meta, dict) else None
        rows.append(m)
        print(f"\n[{label}] {path}")
        print(
            f"  mAP={m['mAP']} 召回={m['recall']} 漏检率={m['miss_rate']} "
            f"图级漏检={m['image_miss_rate']} 过杀={m['fp_per_image']}/图 P={m['precision']} F1={m['f1']}"
        )
        print("  逐类AP:", m.get("per_class_ap"))
        detail["methods"][label] = m  # type: ignore[index]

        if args.sweep:
            curve = threshold_curve(preds, gts, [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0], iou_thr=args.iou, classes=classes)
            save_json(out_dir / f"curve_{Path(path).stem}.json", curve)
            print("  阈值扫描: " + " | ".join(f"{c['thr']}:miss={c['miss_rate']},fp={c['fp_per_image']}" for c in curve))

    print("\n" + md_table(rows))
    save_json(out_dir / "metrics.json", detail)
    (out_dir / "main_table.md").write_text(md_table(rows) + "\n", encoding="utf-8")
    print(f"\n[ok] 表格写入 {out_dir/'main_table.md'}，明细 {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
