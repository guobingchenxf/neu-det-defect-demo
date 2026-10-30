"""延迟基准：分段计时 + 分位数（P50/P95/P99）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from .cnn_model import load_model
from .common import load_annotations, load_splits, resolve_dataset, save_json, set_seed
from .preprocess import TradParams, load_gray, predict_image, run_pipeline
from .slide_infer import infer_image


def pctl(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def summarize(name: str, values: Sequence[float]) -> Dict[str, float]:
    return {
        "name": name,
        "n": len(values),
        "p50": round(pctl(values, 50), 2),
        "p95": round(pctl(values, 95), 2),
        "p99": round(pctl(values, 99), 2),
        "mean": round(float(np.mean(values)), 2),
        "max": round(float(np.max(values)), 2),
    }


def bench_trad(files: Sequence[str], img_dir: Path, params: TradParams, warmup: int = 3) -> List[Dict[str, float]]:
    times: List[float] = []
    for i, name in enumerate(files):
        gray = load_gray(img_dir / name)
        t0 = time.perf_counter()
        predict_image(gray, params)
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:  # 前几张当预热（缓存/分支预测）
            times.append(dt)
    return [summarize("trad_total", times)]


def bench_cnn(
    files: Sequence[str],
    img_dir: Path,
    model: torch.nn.Module,
    patch: int,
    stride: int,
    thr: float,
    warmup: int = 3,
) -> List[Dict[str, float]]:
    total, pre, fwd, post = [], [], [], []
    for i, name in enumerate(files):
        gray = load_gray(img_dir / name)
        t0 = time.perf_counter()
        _, tm, _ = infer_image(model, gray, patch, stride, thr)
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            total.append(dt)
            pre.append(tm["preprocess_ms"])
            fwd.append(tm["forward_ms"])
            post.append(tm["postprocess_ms"])
    return [summarize("cnn_total", total), summarize("cnn_preprocess", pre), summarize("cnn_forward", fwd), summarize("cnn_postprocess", post)]


def thread_sweep(
    files: Sequence[str], img_dir: Path, model: torch.nn.Module, patch: int, stride: int, thr: float
) -> List[Dict[str, float]]:
    out = []
    n = min(len(files), 20)
    for th in (1, 2, 4, 8):
        torch.set_num_threads(th)
        t0 = time.perf_counter()
        for name in files[:n]:
            infer_image(model, load_gray(img_dir / name), patch, stride, thr)
        dt = (time.perf_counter() - t0) / n * 1000
        out.append({"threads": th, "ms_per_image": round(dt, 2)})
        print(f"  threads={th}: {dt:.1f} ms/图")
    torch.set_num_threads(4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="分段延迟基准")
    ap.add_argument("--data-root", default="data/NEU-DET")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--model", default="results/cnn/best.pt")
    ap.add_argument("--params", default="results/trad/best_params.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--thread-sweep", action="store_true")
    ap.add_argument("--out-dir", default="results/latency")
    args = ap.parse_args()

    set_seed(0)
    torch.set_num_threads(max(1, args.threads))
    img_dir, _ann = resolve_dataset(args.data_root)
    splits = load_splits(Path(args.splits))
    files = (splits.get(args.split) or [])[: args.limit]
    if not files:
        raise SystemExit("没有图片可测")

    params = TradParams()
    p = Path(args.params)
    if p.exists():
        import json

        d = json.loads(p.read_text(encoding="utf-8"))
        params = TradParams.from_dict(d.get("params", d))

    print(f"[bench] {len(files)} 张图，patch={args.patch} stride={args.stride} threads={args.threads}")
    trad = bench_trad(files, img_dir, params)
    model = load_model(args.model)
    cnn = bench_cnn(files, img_dir, model, args.patch, args.stride, args.thr)

    rows = trad + cnn
    print(f"\n{'stage':<16}{'P50':>8}{'P95':>8}{'P99':>8}{'mean':>8}")
    for r in rows:
        print(f"{r['name']:<16}{r['p50']:>8}{r['p95']:>8}{r['p99']:>8}{r['mean']:>8}")

    result: Dict[str, object] = {
        "num_images": len(files),
        "patch": args.patch,
        "stride": args.stride,
        "threads": args.threads,
        "rows": rows,
    }
    if args.thread_sweep:
        print("\n[thread sweep] 线程数 vs 延迟")
        result["thread_sweep"] = thread_sweep(files, img_dir, model, args.patch, args.stride, args.thr)

    out = Path(args.out_dir) / "latency.json"
    save_json(out, result)
    print(f"\n[ok] 写入 {out}")


if __name__ == "__main__":
    main()
