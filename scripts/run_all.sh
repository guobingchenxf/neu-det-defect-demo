#!/usr/bin/env bash
# 端到端重跑（合成数据），把每一步的输出都落到 results/logs/。
# 用途：改完代码后确认 README 里的数字还能复现。
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
DATA="${1:-data/FAKE-NEU-DET}"
LOG="${2:-results/logs}"
mkdir -p "$LOG"

# 模块用相对导入（from .common import ...），所以必须 `python -m src.xxx`，不能直接跑文件
run() {
    local tag="$1"
    local mod="$2"
    shift 2
    echo "==> $tag  (src.$mod)"
    "$PY" -u -m "src.$mod" "$@" 2>&1 | tee "$LOG/$tag.log" | tail -n 3
}

run 01_fake_data make_fake_data --out "$DATA" --per-class 12
run 02_splits make_splits --data-root "$DATA" --out data/splits.json
run 03_trad_tune trad_baseline --data-root "$DATA" --mode tune
run 04_trad_eval trad_baseline --data-root "$DATA" --mode eval --split val --sweep
run 05_cnn_train train_cnn --data-root "$DATA" --epochs 15 --lr 5e-4 --threads 4
run 06_slide_infer slide_infer --data-root "$DATA" --mode sweep
run 07_compare evaluate \
    --preds results/trad/preds_val.json results/cnn/preds_val.json \
    --labels trad slide-cnn \
    --data-root "$DATA" --split val --out results/eval
run 08_latency bench_latency --data-root "$DATA" --thread-sweep

echo "==> 09_tests"
"$PY" -u -m tests.test_basic 2>&1 | tee "$LOG/09_tests.log" | tail -n 3
echo "[done] 全部跑完，结果在 results/"
