#!/usr/bin/env bash
set -euo pipefail

# 一键下载官方对比结果并运行完整评测。
OUTPUT_DIR=${1:-"./output/benchmark_run"}

# 下载官方发布的 HELM 输出，以便直接复现实验。
python -m eai_eval.utils.download_utils

# 在所有数据集和评测类型上批量运行结果评测。
eai-eval --mode evaluate_results --all --output-dir "$OUTPUT_DIR"
