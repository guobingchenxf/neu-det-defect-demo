"""逐类召回：六类缺陷分别被哪条路线漏掉。

为什么单独写一个工具而不是直接用 `evaluate`：
`evaluate` 算的是"类别作为标签"的 AP，而本仓库的预测标签统一是 `defect`（先回答"有没有缺陷"）。
要回答"哪一类在拖后腿"，只需要按类把 GT 切开、看每类有多少框被预测命中（IoU=0.5，一对一贪心），
这样不依赖预测的类别标签，也不会因为某类预测标签全错而把 AP 拉成 null。

    python -m tools.per_class_recall --preds results/trad/preds_val.json --label trad
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common import (CLASSES, DEFECT, Box, evaluate_detections, load_annotations,
                        load_preds, load_splits, resolve_dataset, save_json)


def per_class_recall(preds, gts, iou_thr: float = 0.5):
    out = {}
    for cls in CLASSES:
        # 预测标签统一是 defect，所以按类切 GT 时也要把标签改成 defect，否则类不相等永远匹配不上
        sub = {k: [Box(DEFECT, b.x1, b.y1, b.x2, b.y2, b.score) for b in v if b.cls == cls] for k, v in gts.items()}
        n_gt = sum(len(v) for v in sub.values())
        if n_gt == 0:
            out[cls] = None
            continue
        m = evaluate_detections(preds, sub, iou_thr=iou_thr, classes=(DEFECT,))
        out[cls] = {"num_gt": n_gt, "recall": m["recall"], "miss_rate": m["miss_rate"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="results/eval/per_class_recall.json")
    args = ap.parse_args()

    img_dir, ann_dir = resolve_dataset(args.data_root)
    gts_all = load_annotations(img_dir, ann_dir)
    files = load_splits(Path(args.splits)).get(args.split, [])
    gts = {k: gts_all[k] for k in files}
    labels = args.labels or [Path(p).parent.name for p in args.preds]

    table = {}
    for path, label in zip(args.preds, labels):
        preds_all, _meta = load_preds(Path(path))
        preds = {k: preds_all.get(k, []) for k in files}
        table[label] = per_class_recall(preds, gts, args.iou)

    names = list(table)
    print("| 类别 | GT 框数 | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for cls in CLASSES:
        n = next((table[m][cls]["num_gt"] for m in names if table[m][cls]), 0)
        cells = []
        for m in names:
            v = table[m][cls]
            cells.append("—" if v is None else f"{v['recall']:.3f}")
        print(f"| {cls} | {n} | " + " | ".join(cells) + " |")

    save_json(Path(args.out), {"split": args.split, "iou_thr": args.iou, "methods": table})
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
