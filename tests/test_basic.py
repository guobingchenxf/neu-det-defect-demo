"""不依赖 pytest 的最小自测：`python -m tests.test_basic`

覆盖的不是"网络结构"（那种测试意义有限），而是**最容易悄悄错掉、错了还没报错**的地方：
标注坐标口径、切分确定性、IoU/NMS、以及滑窗标签口径（这里错过一次，AUC 直接掉到 0.45）。
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src.common import Box, iou_xyxy, make_stratified_split, nms, parse_voc
from src.patches import _center_dist_to_boxes, _iou_and_cover
from src.preprocess import TradParams, predict_image
from src.slide_infer import _axis_positions, grid_to_boxes
from src.viz import pick_diverse


def test_voc_is_one_based() -> None:
    """VOC 是 1-based 且 xmax 含端点；解析后必须是 0-based 的 [x1, x2)。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.xml"
        root = ET.Element("annotation")
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "200"
        ET.SubElement(size, "height").text = "200"
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "crazing"
        bb = ET.SubElement(obj, "bndbox")
        for k, v in (("xmin", 11), ("ymin", 21), ("xmax", 40), ("ymax", 60)):
            ET.SubElement(bb, k).text = str(v)
        ET.ElementTree(root).write(str(p), encoding="utf-8")

        boxes = parse_voc(p)
        assert len(boxes) == 1, boxes
        b = boxes[0]
        assert (b.x1, b.y1, b.x2, b.y2) == (10.0, 20.0, 40.0, 60.0), (b.x1, b.y1, b.x2, b.y2)
        assert b.cls == "crazing"
    print("  ok  VOC 1-based → 0-based")


def test_iou_and_nms() -> None:
    assert abs(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # 半重叠：交 50，并 150
    assert abs(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9

    boxes = [
        Box("defect", 0, 0, 10, 10, 0.9),
        Box("defect", 1, 1, 11, 11, 0.8),   # 与第一个高度重叠
        Box("defect", 50, 50, 60, 60, 0.7),
    ]
    kept = nms(boxes, iou_thr=0.5)
    assert len(kept) == 2, [b.score for b in kept]
    assert kept[0].score >= kept[1].score
    print("  ok  IoU / NMS")


def test_split_is_deterministic_and_covers_train_classes() -> None:
    gts = {}
    classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    for i in range(60):
        c = classes[i % len(classes)]
        gts[f"{c}_{i:03d}.jpg"] = [Box(c, 1, 1, 20, 20)]
    s1 = make_stratified_split(gts, ratios=(0.6, 0.2, 0.2), seed=7)
    s2 = make_stratified_split(gts, ratios=(0.6, 0.2, 0.2), seed=7)
    assert s1 == s2, "同 seed 切分必须完全一致，否则实验不可复现"
    assert set(s1["train"]) | set(s1["tune"]) | set(s1["val"]) == set(gts)
    assert not (set(s1["train"]) & set(s1["val"]))
    for c in classes:
        assert any(n.startswith(c) for n in s1["train"]), f"train 里没有 {c}，分层失效"
    print("  ok  分层切分（确定性 + 每类都有）")


def test_patch_label_uses_center_not_iou() -> None:
    """滑窗正样本看"窗口中心点是否落在缺陷内"，不是 IoU，也不是覆盖率。

    这两个坑各毁过一次实验：IoU → AUC 0.45；覆盖率 → 框全大一圈、mAP 0.11。
    写成测试，防止以后又改回去。
    """
    big = [Box("patches", 0, 0, 120, 120)]
    # 中心在缺陷内部 → 正（尽管窗口只有 64，IoU 很低）
    assert _center_dist_to_boxes(60, 60, big) == 0.0
    iou, _cover = _iou_and_cover(28, 28, 92, 92, big)
    assert iou < 0.35, iou

    # 中心在缺陷外 → 不能算正样本
    assert _center_dist_to_boxes(160, 160, big) > 64 * 0.5
    print("  ok  滑窗标签口径（中心点）")


def test_axis_positions_covers_region() -> None:
    """两阶段的细搜区域必须铺满到边界，否则贴着粗框边缘的缺陷永远搜不到。"""
    pos = _axis_positions(10, 90, patch=64, step=8, limit=200)
    assert pos[0] == 10, pos[0]
    assert pos[-1] + 64 >= 90, (pos[-1], pos[-1] + 64)
    assert all(p + 64 <= 200 for p in pos), pos
    assert all(b - a == 8 for a, b in zip(pos, pos[1:-1])), pos
    # 区域比 patch 还小时，退化成单窗且不越界
    tiny = _axis_positions(190, 200, patch=64, step=8, limit=200)
    assert tiny == [136], tiny
    print("  ok  细搜区域铺满（_axis_positions）")


def test_grid_to_boxes_is_not_patch_sized() -> None:
    """一个亮格子只能出半个 patch 大的框，不能出整张 patch 的框。"""
    patch, stride = 64, 32
    xs = [0, 32, 64, 96, 128]
    ys = [0, 32, 64, 96, 128]
    grid = np.zeros((len(ys), len(xs)), np.float32)
    grid[2, 2] = 0.9  # 只有一格亮
    boxes = grid_to_boxes(grid, xs, ys, patch, thr=0.5)
    assert len(boxes) == 1, boxes
    b = boxes[0]
    assert (b.x2 - b.x1) <= patch, (b.x2 - b.x1)
    assert (b.x2 - b.x1) == patch // 2, (b.x2 - b.x1)
    assert abs(b.x1 - (xs[2] + patch // 4)) < 1e-6, b.x1
    print("  ok  热力图出框尺寸（格心 ± stride/2，不是整张 patch）")


def test_trad_finds_a_blob() -> None:
    img = np.full((200, 200), 160, np.uint8)
    img[80:120, 80:130] = 40  # 一块明显的暗缺陷
    p = TradParams(bin_method="percentile", pct=99.0, min_area=50, min_fill=0.2, merge_k=1)
    boxes = predict_image(img, p)
    assert boxes, "很明显的一块缺陷都找不到，说明预处理链写错了"
    best = max(boxes, key=lambda b: b.score)
    assert best.x1 < 130 and best.x2 > 80 and best.y1 < 120 and best.y2 > 80, best.to_list()
    print(f"  ok  传统基线能找到合成块（{len(boxes)} 个框）")


def test_pick_diverse_covers_each_class() -> None:
    """draw 模式必须每类各挑一张：只看前 6 张会全是 crazing，容易误判"整份数据都这样"。"""
    files = [f"img{i}.jpg" for i in range(8)]
    gts = {f: [Box("crazing" if i % 2 else "scratches", 1, 1, 20, 20)] for i, f in enumerate(files)}
    picked = pick_diverse(files, gts, 4)
    assert len(picked) == 4, picked
    assert len(set(picked)) == 4, "不能重复挑同一张"
    assert {gts[n][0].cls for n in picked[:2]} == {"crazing", "scratches"}, picked
    print("  ok  draw 按类挑图（pick_diverse）")


def test_per_class_recall_uses_defect_label() -> None:
    """逐类召回必须把该类 GT 的标签改成 defect 再按类别无关模式评估。

    坑：预测标签统一是 defect，如果按 `classes=("inclusion",)` 去算，
    类别不一致 → 一个都匹配不上 → 召回静默变成 0（我第一次跑逐类表就是这样，六类全 0）。
    """
    from src.common import DEFECT, evaluate_detections

    preds = {"a.jpg": [Box(DEFECT, 10, 10, 50, 50, 0.9)]}
    gts = {"a.jpg": [Box("inclusion", 10, 10, 50, 50, 1.0)]}
    sub = {k: [Box(DEFECT, b.x1, b.y1, b.x2, b.y2, b.score) for b in v] for k, v in gts.items()}
    assert evaluate_detections(preds, sub, classes=(DEFECT,))["recall"] == 1.0
    assert evaluate_detections(preds, gts, classes=("inclusion",))["recall"] == 0.0
    print("  ok  逐类召回的标签映射（不改标签会静默算成 0）")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n[ok] {len(fns)} 项自测全部通过")
