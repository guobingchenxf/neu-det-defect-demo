"""共用工具：数据集定位、VOC 标注解析、分层切分、IoU / NMS、检测评估。

设计原则
1. 评估口径与训练解耦 —— 传统基线和滑窗 CNN 共用同一套评估代码，
   保证"两条路线用同一把尺子量"。
2. 切分一次生成、落盘复用 —— 避免"每次跑随机切分"导致指标不可复现。
3. 类别无关模式 —— 传统管线判类不可靠，所以主表按"defect"单类评估定位能力，
   判类作为单独一组实验（--classify），不把定位和分类混在一个数字里。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# NEU-DET 的 6 类（目录/标注里的 class 名，大小写以 XML 为准，匹配时不区分）
CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
DEFECT = "defect"  # 类别无关模式下的单一类别名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class Box:
    """xyxy 坐标，0-based，浮点。score 只有预测框才有意义。"""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2, self.score]

    def clipped(self, w: int, h: int) -> "Box":
        return Box(
            self.cls,
            float(np.clip(self.x1, 0, w)),
            float(np.clip(self.y1, 0, h)),
            float(np.clip(self.x2, 0, w)),
            float(np.clip(self.y2, 0, h)),
            self.score,
        )


def norm_cls(name: str) -> str:
    """把 XML/文件名里的类名归一化：小写、空格/下划线统一。"""
    return name.strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# 数据集定位（兼容几种常见的 NEU-DET 目录摆放）
# --------------------------------------------------------------------------- #
def resolve_dataset(root: str | Path) -> Tuple[Path, Path]:
    """返回 (images_dir, annotations_dir)。找不到就抛错并给出提示。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    img_dir: Optional[Path] = None
    ann_dir: Optional[Path] = None
    for base, dirs, files in os.walk(root):
        depth = len(Path(base).relative_to(root).parts)
        if depth >= 3:
            dirs[:] = []
            continue
        for d in list(dirs):
            low = d.lower()
            if img_dir is None and low in {"images", "image", "imgs", "img"}:
                img_dir = Path(base) / d
            if ann_dir is None and low in {"annotations", "annotation", "anns", "labels", "xml", "ground-truth"}:
                ann_dir = Path(base) / d
        if img_dir is None and any(f.lower().endswith(".xml") for f in files):
            # 图片与 XML 混在同一层目录
            if any(Path(f).suffix.lower() in IMG_EXTS for f in files):
                img_dir, ann_dir = Path(base), Path(base)

    if img_dir is None:
        raise FileNotFoundError(
            f"在 {root} 下没找到图片目录（期望名为 IMAGES / images）。"
            "请把 NEU-DET 的 IMAGES 与 ANNOTATIONS 放进去，或运行 python -m src.make_fake_data 生成合成数据试跑。"
        )
    if ann_dir is None:
        raise FileNotFoundError(f"在 {root} 下没找到 XML 标注目录（期望名为 ANNOTATIONS）。")
    return img_dir, ann_dir


def list_images(img_dir: Path) -> List[Path]:
    files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not files:
        raise FileNotFoundError(f"{img_dir} 下没有图片")
    return sorted(files)


# --------------------------------------------------------------------------- #
# VOC XML 标注
# --------------------------------------------------------------------------- #
def parse_voc(xml_path: Path, w: Optional[int] = None, h: Optional[int] = None) -> List[Box]:
    import xml.etree.ElementTree as ET

    boxes: List[Box] = []
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    for obj in root.iter("object"):
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xs = [bb.find(k) for k in ("xmin", "xmax")]
        ys = [bb.find(k) for k in ("ymin", "ymax")]
        if any(v is None or v.text is None for v in xs + ys):
            continue
        x1, x2 = float(xs[0].text), float(xs[1].text)
        y1, y2 = float(ys[0].text), float(ys[1].text)
        # VOC 的 bndbox 是 1-based 且 xmax/ymax 含端点：
        #   像素区间 [xmin-1, xmax-1]（闭） == 半开区间 [xmin-1, xmax)
        # 所以只把 xmin 减 1、xmax 保持不动。早先两边都减 1 会让每个框小 1 像素，
        # 对 20 像素的小目标等于白白损失 10% 的 IoU —— 这种错不会报错，只会让指标变差。
        b = Box(norm_cls(name_el.text), x1 - 1, y1 - 1, x2, y2, 1.0)
        if b.area() <= 0:
            continue
        if w is not None and h is not None:
            b = b.clipped(w, h)
        if b.area() <= 0:
            continue
        boxes.append(b)
    return boxes


def load_annotations(img_dir: Path, ann_dir: Path) -> Dict[str, List[Box]]:
    """读全部标注：{图片文件名: [Box]}。没有 XML 的图片记为"无缺陷"（重要，否则 FP 统计失真）。"""
    gts: Dict[str, List[Box]] = {}
    for img in list_images(img_dir):
        xml = ann_dir / f"{img.stem}.xml"
        gts[img.name] = parse_voc(xml) if xml.exists() else []
    return gts


# --------------------------------------------------------------------------- #
# 分层切分（按图片的主类别分层；一次生成、落盘复用）
# --------------------------------------------------------------------------- #
def image_label(boxes: Sequence[Box]) -> str:
    if not boxes:
        return "__empty__"
    counts: Dict[str, int] = {}
    for b in boxes:
        counts[b.cls] = counts.get(b.cls, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def make_stratified_split(
    gts: Dict[str, List[Box]],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> Dict[str, List[str]]:
    """train / tune / val 三份。

    tune 单独留出来的原因：传统管线的参数要选，如果在 val 上选就是"用测试集调参"，
    报出来的指标会虚高。所以参数只在 tune 上搜，val 只用来报最终数字。
    """
    rng = random.Random(seed)
    by_cls: Dict[str, List[str]] = {}
    for name, boxes in gts.items():
        by_cls.setdefault(image_label(boxes), []).append(name)

    splits: Dict[str, List[str]] = {"train": [], "tune": [], "val": []}
    for cls, names in sorted(by_cls.items()):
        names = sorted(names)
        rng.shuffle(names)
        n = len(names)
        n_val = max(1, int(round(n * ratios[2]))) if n >= 4 else max(0, n // 5)
        n_tune = max(1, int(round(n * ratios[1]))) if n >= 4 else 0
        n_train = n - n_val - n_tune
        if n_train <= 0:  # 极小类别保底
            n_train, n_tune, n_val = max(1, n - 1), 0, min(1, n - 1)
        splits["train"] += names[:n_train]
        splits["tune"] += names[n_train : n_train + n_tune]
        splits["val"] += names[n_train + n_tune :]
    for k in splits:
        splits[k] = sorted(splits[k])
    return splits


def save_splits(splits: Dict[str, List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")


def load_splits(path: Path) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# IoU / NMS / 匹配
# --------------------------------------------------------------------------- #
def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(gt_boxes: Sequence[Sequence[float]], pr_boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """(n_gt, n_pr) 的 IoU 矩阵。"""
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pr_boxes)), dtype=np.float32)
    g = np.asarray([b[:4] for b in gt_boxes], dtype=np.float32)
    p = np.asarray([b[:4] for b in pr_boxes], dtype=np.float32)
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0])
    iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2])
    iy2 = np.minimum(g[:, None, 3], p[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    ga = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    pa = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    union = ga[:, None] + pa[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def nms(boxes: Sequence[Box], iou_thr: float = 0.5) -> List[Box]:
    if not boxes:
        return []
    order = np.argsort([-b.score for b in boxes])
    keep: List[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        ious = np.asarray([iou_xyxy(boxes[i].xyxy(), boxes[j].xyxy()) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_thr]
    return [boxes[i] for i in keep]


def match(
    preds: Sequence[Box], gts: Sequence[Box], iou_thr: float = 0.5
) -> Tuple[List[int], List[int]]:
    """贪心一对一匹配（按预测分数降序）。

    返回 (tp 下标列表, fp 下标列表)。一个 GT 只允许被匹配一次。
    """
    if not preds or not gts:
        return [], list(range(len(preds)))
    order = np.argsort([-b.score for b in preds])
    used = [False] * len(gts)
    tp: List[int] = []
    fp: List[int] = []
    for pi in order:
        pi = int(pi)
        best_iou, best_gi = 0.0, -1
        for gi, g in enumerate(gts):
            if used[gi]:
                continue
            v = iou_xyxy(preds[pi].xyxy(), g.xyxy())
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_thr and best_gi >= 0:
            used[best_gi] = True
            tp.append(pi)
        else:
            fp.append(pi)
    return tp, fp


# --------------------------------------------------------------------------- #
# 评估：AP / mAP / 漏检率 / 过杀率
# --------------------------------------------------------------------------- #
def _average_precision(scores: Sequence[float], tps: Sequence[bool], n_gt: int) -> float:
    """全点插值 AP（VOC2010+ / COCO 风格）。"""
    if n_gt <= 0 or len(scores) == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray(tps, dtype=np.float64)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    rec = tp_cum / max(n_gt, 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detections(
    preds_by_img: Dict[str, List[Box]],
    gts_by_img: Dict[str, List[Box]],
    iou_thr: float = 0.5,
    classes: Sequence[str] = (DEFECT,),
    score_thr: float = 0.0,
) -> Dict[str, object]:
    """返回一个可直接 json 化的指标字典（两条路线共用同一把尺子）。

    指标口径（写进报告，避免"数字说不清"）：
    - AP / mAP：逐类匹配（IoU>=iou_thr 且类别一致才算 TP），全点插值，GT 数做归一。
    - recall / miss_rate：类别感知的框级召回，漏检率 = 1 - recall。
    - image_miss_rate：有缺陷的图里一个都没检出的比例；这里是**类别无关**匹配
      （为了和产线"这张图有没有被报出来"的口径一致），与上面的类内 recall 不同源。
    - fp_per_image：全部预测里未匹配上的框数 / 图片数，即"过杀"，分母是全部图片。
    """
    names = sorted({n for n in list(preds_by_img) + list(gts_by_img)})
    scores: Dict[str, List[float]] = {c: [] for c in classes}
    tps: Dict[str, List[bool]] = {c: [] for c in classes}
    n_gt_c: Dict[str, int] = {c: 0 for c in classes}
    n_tp_c: Dict[str, int] = {c: 0 for c in classes}

    n_pred = 0
    fp_total = 0
    img_with_gt = 0
    img_missed = 0

    for name in names:
        gts_all = list(gts_by_img.get(name, []))
        preds_all = [b for b in preds_by_img.get(name, []) if b.score >= score_thr]

        if tuple(classes) == (DEFECT,):  # 类别无关模式：全部视作同一类
            gts_all = [Box(DEFECT, *b.xyxy(), 1.0) for b in gts_all]
            preds_all = [Box(DEFECT, *b.xyxy(), b.score) for b in preds_all]

        n_pred += len(preds_all)

        for c in classes:
            gts_c = [g for g in gts_all if g.cls == c]
            preds_c = [p for p in preds_all if p.cls == c]
            n_gt_c[c] += len(gts_c)
            tp_idx, fp_idx = match(preds_c, gts_c, iou_thr)
            n_tp_c[c] += len(tp_idx)
            fp_total += len(fp_idx)
            for pi in tp_idx:
                scores[c].append(float(preds_c[pi].score))
                tps[c].append(True)
            for pi in fp_idx:
                scores[c].append(float(preds_c[pi].score))
                tps[c].append(False)

        tp_any, _ = match(preds_all, gts_all, iou_thr)
        if gts_all:
            img_with_gt += 1
            if len(tp_any) == 0:
                img_missed += 1

    per_class_ap: Dict[str, Optional[float]] = {}
    for c in classes:
        ap = _average_precision(scores[c], tps[c], n_gt_c.get(c, 0))
        per_class_ap[c] = None if np.isnan(ap) else round(ap, 4)

    valid_aps = [v for v in per_class_ap.values() if v is not None]
    n_gt_total = sum(n_gt_c.values())
    n_tp_total = sum(n_tp_c.values())
    recall = (n_tp_total / n_gt_total) if n_gt_total else float("nan")

    return {
        "iou_thr": iou_thr,
        "score_thr": score_thr,
        "classes": list(classes),
        "per_class_ap": per_class_ap,
        "mAP": round(float(np.mean(valid_aps)), 4) if valid_aps else None,
        "recall": None if np.isnan(recall) else round(recall, 4),
        "miss_rate": None if np.isnan(recall) else round(1 - recall, 4),
        "image_miss_rate": round(img_missed / img_with_gt, 4) if img_with_gt else None,
        "fp_per_image": round(fp_total / max(len(names), 1), 4),
        "num_pred": n_pred,
        "num_gt": n_gt_total,
        "num_images": len(names),
    }


def threshold_curve(
    preds_by_img: Dict[str, List[Box]],
    gts_by_img: Dict[str, List[Box]],
    thrs: Iterable[float],
    iou_thr: float = 0.5,
    classes: Sequence[str] = (DEFECT,),
) -> List[Dict[str, float]]:
    """阈值扫描：交付时真正要交的是"漏检率-过杀率"曲线上的一 个工作点。"""
    curve = []
    for t in thrs:
        m = evaluate_detections(preds_by_img, gts_by_img, iou_thr=iou_thr, classes=classes, score_thr=float(t))
        curve.append(
            {
                "thr": round(float(t), 4),
                "miss_rate": m["miss_rate"],
                "recall": m["recall"],
                "fp_per_image": m["fp_per_image"],
                "mAP": m["mAP"],
                "num_pred": m["num_pred"],
            }
        )
    return curve


# --------------------------------------------------------------------------- #
# 预测结果落盘格式（两条路线共用）
# --------------------------------------------------------------------------- #
def save_preds(
    path: Path,
    preds_by_img: Dict[str, List[Box]],
    meta: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "meta": meta or {},
        "images": {k: [b.to_list() + [b.cls] for b in v] for k, v in sorted(preds_by_img.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_preds(path: Path) -> Tuple[Dict[str, List[Box]], Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, List[Box]] = {}
    for name, rows in payload.get("images", {}).items():
        out[name] = [Box(cls=r[5], x1=r[0], y1=r[1], x2=r[2], y2=r[3], score=r[4]) for r in rows]
    return out, payload.get("meta", {})


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
