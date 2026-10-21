"""真实数据（NEU-DET）分阶段重跑。

合成数据的 run_all.py 不适合真实数据：tune 要重搜、CNN 训练要限时。
这个脚本按 tag 跑指定阶段，每阶段单独落日志，方便中途断点续跑：

    python scripts/run_real.py trad          # 传统基线：tune + eval
    python scripts/run_real.py cnn           # 训练滑窗 CNN
    python scripts/run_real.py infer         # stride 16 / 32 两套推理
    python scripts/run_real.py eval          # 统一对比表 + 逐类表
    python scripts/run_real.py lat           # 延迟与线程扫描
    python scripts/run_real.py tests         # 单测
    python scripts/run_real.py all           # 以上全部（不含 trad tune，那个单独跑）

不传参数打印可用 tag。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs"
DATA = "data/NEU-DET"

STEPS = {
    "trad": [
        ("real_trad_tune", "src.trad_baseline", ["--data-root", DATA, "--mode", "tune"]),
        ("real_trad_eval", "src.trad_baseline",
         ["--data-root", DATA, "--mode", "eval", "--split", "val", "--sweep"]),
    ],
    "cnn": [
        # 真实数据 1258 张训练图，全量采样一轮要 20 分钟以上；先限 500 张，够用且能在 CPU 跑完。
        ("real_cnn_train", "src.train_cnn",
         ["--data-root", DATA, "--epochs", "12", "--lr", "5e-4",
          "--max-train-images", "500", "--threads", "4"]),
    ],
    "infer": [
        ("real_infer_s16", "src.slide_infer",
         ["--data-root", DATA, "--mode", "eval", "--stride", "16", "--out-dir", "results/cnn_s16"]),
        ("real_infer_s32", "src.slide_infer",
         ["--data-root", DATA, "--mode", "eval", "--stride", "32", "--out-dir", "results/cnn_s32"]),
    ],
    "eval": [
        ("real_compare", "src.evaluate",
         ["--preds", "results/trad/preds_val.json",
          "results/cnn_s16/preds_val.json", "results/cnn_s32/preds_val.json",
          "--labels", "trad", "slide16", "slide32",
          "--data-root", DATA, "--split", "val", "--out", "results/eval"]),
        ("real_compare_cls", "src.evaluate",
         ["--preds", "results/trad/preds_val.json",
          "results/cnn_s16/preds_val.json", "results/cnn_s32/preds_val.json",
          "--labels", "trad", "slide16", "slide32", "--classes", "all",
          "--data-root", DATA, "--split", "val", "--out", "results/eval_cls"]),
    ],
    "lat": [
        ("real_latency", "src.bench_latency",
         ["--data-root", DATA, "--thread-sweep", "--stride", "16"]),
    ],
    "tests": [("real_tests", "tests.test_basic", [])],
}

ORDER = ["trad", "cnn", "infer", "eval", "lat", "tests"]


def run_step(tag: str, mod: str, args: list[str]) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"{tag}.log"
    t0 = time.time()
    print(f"==> {tag}  ({mod})", flush=True)
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, "-u", "-m", mod, *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        fh.write(proc.stdout or "")
    dt = time.time() - t0
    tail = (proc.stdout or "").strip().splitlines()[-4:]
    for line in tail:
        print("   " + line)
    print(f"   [{dt:.0f}s] 日志 {log_path.name}", flush=True)
    return proc.returncode


def main() -> int:
    tags = sys.argv[1:]
    if not tags:
        print("可用 tag:", ", ".join(ORDER), "| all")
        return 0
    if tags == ["all"]:
        tags = ["trad", "cnn", "infer", "eval", "lat", "tests"]
    for tag in tags:
        if tag not in STEPS:
            print(f"[skip] 未知 tag: {tag}")
            continue
        for step in STEPS[tag]:
            rc = run_step(*step)
            if rc != 0:
                print(f"[FAIL] {step[0]} 退出码 {rc}")
                return rc
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
