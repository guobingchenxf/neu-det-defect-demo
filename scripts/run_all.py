"""端到端重跑（跨平台，不依赖 bash）。

    python scripts/run_all.py                 # 合成数据全流程
    python scripts/run_all.py data/NEU-DET    # 真实数据全流程

每一步都单独落日志到 results/logs/NN_xxx.log，方便回头查是哪一步变了。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/FAKE-NEU-DET"
LOG = ROOT / "results" / "logs"

STEPS = [
    ("01_fake_data", "src.make_fake_data", ["--out", DATA, "--per-class", "12"], True),
    ("02_splits", "src.make_splits", ["--data-root", DATA, "--out", "data/splits.json"], False),
    ("03_trad_tune", "src.trad_baseline", ["--data-root", DATA, "--mode", "tune"], False),
    ("04_trad_eval", "src.trad_baseline", ["--data-root", DATA, "--mode", "eval", "--split", "val", "--sweep"], False),
    ("05_cnn_train", "src.train_cnn", ["--data-root", DATA, "--epochs", "15", "--lr", "5e-4", "--threads", "4"], False),
    ("06_slide_infer", "src.slide_infer", ["--data-root", DATA, "--mode", "sweep"], False),
    (
        "07_compare",
        "src.evaluate",
        [
            "--preds", "results/trad/preds_val.json", "results/cnn/preds_val.json",
            "--labels", "trad", "slide-cnn",
            "--data-root", DATA, "--split", "val", "--out", "results/eval",
        ],
        False,
    ),
    ("08_latency", "src.bench_latency", ["--data-root", DATA, "--thread-sweep"], False),
    ("09_tests", "tests.test_basic", [], False),
]


def main() -> int:
    if DATA == "data/FAKE-NEU-DET" and not (ROOT / DATA).exists():
        pass  # 第一步会生成
    LOG.mkdir(parents=True, exist_ok=True)
    for tag, mod, args, only_if_missing in STEPS:
        if only_if_missing and (ROOT / DATA).exists():
            print(f"==> {tag} 跳过（{DATA} 已存在）")
            continue
        print(f"==> {tag}  ({mod})", flush=True)
        log_path = LOG / f"{tag}.log"
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
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        for line in tail:
            print("   " + line)
        if proc.returncode != 0:
            print(f"[FAIL] {tag} 退出码 {proc.returncode}，日志见 {log_path}")
            return proc.returncode
    print("[done] 全部跑完，结果在 results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
