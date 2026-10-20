#!/usr/bin/env bash
# 一键跑完两条路线（传统基线 + 滑窗小 CNN）并出对比表。
# 用法：
#   bash scripts/quickstart.sh              # 没有数据就自动生成合成数据
#   bash scripts/quickstart.sh data/NEU-DET # 用真实 NEU-DET
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
DATA="${1:-data/FAKE-NEU-DET}"

if [ ! -d "$DATA" ]; then
    echo "==> 没找到 $DATA，先生成合成数据"
    "$PY" -m src.make_fake_data --out "$DATA" --per-class 12
fi

echo "==> 1/6 分层切分（按『图的主缺陷类』分层，不是按框）"
"$PY" -m src.make_splits --data-root "$DATA" --out data/splits.json

echo "==> 2/6 传统基线：只在 tune 集上网格搜索参数（不碰 val）"
"$PY" -m src.trad_baseline --data-root "$DATA" --mode tune

echo "==> 3/6 传统基线：val 上评估 + 阈值扫描"
"$PY" -m src.trad_baseline --data-root "$DATA" --mode eval --split val --sweep

echo "==> 4/6 滑窗 + 小 CNN：训练 patch 分类器（CPU 也能跑）"
"$PY" -m src.train_cnn --data-root "$DATA" --epochs 12 --threads 4

echo "==> 5/6 滑窗 + 小 CNN：热力图 → 连通域 → NMS"
"$PY" -m src.slide_infer --data-root "$DATA" --mode sweep

echo "==> 6/6 两条路线放同一张表"
"$PY" -m src.evaluate \
    --preds results/trad/preds_val.json results/cnn/preds_val.json \
    --labels "传统基线" "滑窗+小CNN" \
    --data-root "$DATA" --split val --out results/eval

echo
echo "完成。表格在 results/eval/main_table.md，逐图预测在 results/*/preds_val.json"
echo "可视化： python -m src.trad_baseline --mode draw   /   python -m src.slide_infer --mode draw"
