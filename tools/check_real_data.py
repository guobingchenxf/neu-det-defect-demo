"""体检：真实数据集能否被 src/common.py 正确解析。

只读，不写任何结果文件。用法：
    python tools/check_real_data.py --data-root data/NEU-DET
"""
import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import (  # noqa: E402
    CLASSES, resolve_dataset, list_images, load_annotations, image_label,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/NEU-DET")
    args = ap.parse_args()
    root = args.data_root

    img_dir, ann_dir = resolve_dataset(root)
    print("[resolve] img_dir =", img_dir)
    print("[resolve] ann_dir =", ann_dir)

    # 注意：load_annotations 的 key 是图片文件名（str），list_images 返回的是 Path，
    # 这里统一换成文件名再查，别拿 Path 去查（查不到会静默当成"没有标注"）。
    imgs = [p.name for p in list_images(img_dir)]
    gts = load_annotations(img_dir, ann_dir)
    print("[files] images = %d, annotations = %d" % (len(imgs), len(gts)))

    missing = [n for n in imgs if n not in gts]
    empty = [n for n in imgs if n in gts and not gts[n]]
    print("[pairing] image 缺 xml = %d, xml 无框 = %d" % (len(missing), len(empty)))
    if missing[:5]:
        print("  缺 xml 示例:", missing[:5])
    if empty[:5]:
        print("  无框示例:", empty[:5])

    cls_cnt = collections.Counter()
    for boxes in gts.values():
        for b in boxes:
            cls_cnt[b.cls] += 1
    print("[classes] 框数:", dict(sorted(cls_cnt.items())))
    print("[classes] 未在 CLASSES 里的:", sorted(set(cls_cnt) - set(CLASSES)))

    img_cnt = collections.Counter(image_label(gts.get(n, [])) for n in imgs)
    print("[classes] 图片数:", dict(sorted(img_cnt.items())))

    nbox = collections.Counter(len(gts.get(n, [])) for n in imgs)
    print("[boxes/img] 直方图:", dict(sorted(nbox.items())))

    allb = [b for v in gts.values() for b in v]
    if not allb:
        print("[FATAL] 没有解析出任何框")
        return 1
    ws = sorted(b.x2 - b.x1 for b in allb)
    hs = sorted(b.y2 - b.y1 for b in allb)
    ar = sorted((b.x2 - b.x1) / float(b.y2 - b.y1) for b in allb)
    areas = sorted(b.area() for b in allb)
    print("[boxes] total = %d" % len(allb))
    print("[boxes] w  min/med/max = %d / %.1f / %d" % (ws[0], statistics.median(ws), ws[-1]))
    print("[boxes] h  min/med/max = %d / %.1f / %d" % (hs[0], statistics.median(hs), hs[-1]))
    print("[boxes] ar min/med/max = %.2f / %.2f / %.2f" % (ar[0], statistics.median(ar), ar[-1]))
    print("[boxes] area 中位数 %.0f px, <32px 边长(both) 占比 %.1f%%"
          % (statistics.median(areas),
             100.0 * sum(1 for b in allb if (b.x2 - b.x1) < 32 and (b.y2 - b.y1) < 32) / len(allb)))

    bad = [b for b in allb
           if b.x1 < 0 or b.y1 < 0 or b.x2 > 200 or b.y2 > 200 or b.x2 <= b.x1 or b.y2 <= b.y1]
    print("[boxes] 越界/非法 = %d" % len(bad))
    if bad:
        print("  示例:", [(b.cls, b.x1, b.y1, b.x2, b.y2) for b in bad[:5]])

    # 小框比例决定 stride 32 能不能达标：40px 框在 stride 32 网格上只剩 1 格
    small = sum(1 for b in allb if max(b.x2 - b.x1, b.y2 - b.y1) < 48)
    print("[risk] 最长边 <48px 的框占比 %.1f%%（这些框在 stride=32 上最多命中 1~2 格）"
          % (100.0 * small / len(allb)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
