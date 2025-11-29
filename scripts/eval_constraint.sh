#!/usr/bin/env bash
# Run baseline and constraint-augmented evaluations for subgoal decomposition
# and generate comparison plots.
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Usage: scripts/eval_constraint.sh <llm_response_path> [output_dir]

Arguments:
  llm_response_path   Path to the LLM response JSON for subgoal decomposition.
  output_dir          Directory to store evaluation outputs (default: ./output/constraint_eval)

Environment variables:
  RUNS                Number of repeated runs for each pipeline (default: 3)
  MIN_STEPS           Minimum subgoal count when sampling tasks (default: 10)
  MAX_STEPS           Maximum subgoal count when sampling tasks (default: 30)
  CONSTRAINT_MODE     Constraint mode passed to the constraint-augmented pipeline (default: affordance)
  SAMPLE_SIZE         Optional cap on sampled tasks (unset = all matching tasks)
  SEED                Base random seed for sampling and runs (default: 0)
USAGE
}

if [[ ${1-} == "-h" || ${1-} == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Error: llm_response_path is required" >&2
  print_usage
  exit 1
fi

LLM_RESPONSE_PATH="$1"
OUTPUT_DIR="${2:-./output/constraint_eval}"
RUNS="${RUNS:-3}"
MIN_STEPS="${MIN_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:-30}"
CONSTRAINT_MODE="${CONSTRAINT_MODE:-affordance}"
SEED="${SEED:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"

mkdir -p "$OUTPUT_DIR"

EVAL_CMD=(
  python -m behavior_eval.evaluation.subgoal_decomposition.scripts.evaluate_results
  evaluate_pipelines
  --llm_response_path "$LLM_RESPONSE_PATH"
  --result_dir "$OUTPUT_DIR"
  --runs "$RUNS"
  --min_steps "$MIN_STEPS"
  --max_steps "$MAX_STEPS"
  --constraint_mode "$CONSTRAINT_MODE"
  --seed "$SEED"
)

if [[ -n "$SAMPLE_SIZE" ]]; then
  EVAL_CMD+=(--sample_size "$SAMPLE_SIZE")
fi

echo "Running baseline vs constraint comparison..."
"${EVAL_CMD[@]}"

COMPARISON_JSON="$OUTPUT_DIR/comparison/pipeline_comparison.json"
PLOT_DIR="$OUTPUT_DIR/plots"

if [[ ! -f "$COMPARISON_JSON" ]]; then
  echo "Expected comparison file not found at $COMPARISON_JSON" >&2
  exit 1
fi

echo "Generating plots..."
python graph.py --comparison-path "$COMPARISON_JSON" --output-dir "$PLOT_DIR"

echo "All artifacts saved to $OUTPUT_DIR"
