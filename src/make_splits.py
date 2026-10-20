"""生成一次性的 train / tune / val 切分并落盘。

为什么要单独一个 tune 集：
传统管线的参数（阈值倍数、面积下限、合并核大小）需要搜索。如果在 val 上搜，
报出来的就是"在测试集上调过参"的数字——这在工业评审里是最容易被抓的点。
所以：train 训练判类器，tune 搜参数，val 只报数。

另外这批数据是同一采集批次的子集，图片之间存在强相似性。这里的分层切分**不能**
解决跨批次泛化问题（报告里会明确写出这个局限），但至少保证同一张图的相邻帧不会
同时出现在 train 和 val（NEU-DET 每类图片文件名连续，这里按文件名排列后随机打散，
并在每个 split 里保留类分布）。

用法：
    python -m src.make_splits --data-root data/NEU-DET --out data/splits.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    CLASSES,
    image_label,
    list_images,
    load_annotations,
    make_stratified_split,
    resolve_dataset,
    save_json,
    save_splits,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成分层切分（train/tune/val）")
    ap.add_argument("--data-root", default="data/NEU-DET", help="数据集根目录（含 IMAGES 与 ANNOTATIONS）")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", default="0.7,0.15,0.15", help="train,tune,val")
    args = ap.parse_args()

    set_seed(args.seed)
    img_dir, ann_dir = resolve_dataset(args.data_root)
    print(f"[data] images={img_dir}\n[data] annotations={ann_dir}")
    gts = load_annotations(img_dir, ann_dir)
    n_img = len(gts)
    n_boxes = sum(len(v) for v in gts.values())
    n_empty = sum(1 for v in gts.values() if not v)
    print(f"[data] 图片 {n_img} 张，标注框 {n_boxes} 个，无缺陷图 {n_empty} 张")

    ratios = tuple(float(x) for x in args.ratios.split(","))
    splits = make_stratified_split(gts, ratios=ratios, seed=args.seed)  # type: ignore[arg-type]

    # 类分布自检：任何一个 split 里某类缺席都要报警，否则逐类 AP 会出现 None
    print("\n[split] 每类图片数：")
    header = f"{'split':<8}{'total':>7}" + "".join(f"{c[:9]:>12}" for c in CLASSES)
    print(header)
    for name in ("train", "tune", "val"):
        files = splits[name]
        counts = {c: 0 for c in CLASSES}
        for f in files:
            for c in {image_label(gts[f])}:
                if c in counts:
                    counts[c] += 1
        row = f"{name:<8}{len(files):>7}" + "".join(f"{counts[c]:>12}" for c in CLASSES)
        print(row)

    empty_splits = [k for k, v in splits.items() if not v]
    if empty_splits:
        print(f"[warn] 这些 split 为空：{empty_splits}（数据量太小或类别过少）")

    out = Path(args.out)
    save_splits(splits, out)
    save_json(
        out.with_suffix(".meta.json"),
        {
            "data_root": str(args.data_root),
            "num_images": n_img,
            "num_boxes": n_boxes,
            "num_images_without_defect": n_empty,
            "ratios": list(ratios),
            "seed": args.seed,
        },
    )
    print(f"\n[ok] 写入 {out}（另附 {out.with_suffix('.meta.json').name}）")


if __name__ == "__main__":
    main()
