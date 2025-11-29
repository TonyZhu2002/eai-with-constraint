#!/usr/bin/env python3
"""Plot baseline vs constraint evaluation results for subgoal decomposition.

This script expects the comparison JSON produced by
`behavior_eval.evaluation.subgoal_decomposition.scripts.evaluate_results` via
its ``evaluate_pipelines`` entrypoint. By default it reads
``./output/constraint_eval/comparison/pipeline_comparison.json`` and saves all
plots to ``./output/constraint_eval/plots``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


MetricName = str
PipelineName = str


def _load_comparison(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Comparison file not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_metric_bars(
    pipelines: Dict[PipelineName, Dict],
    metrics: List[Tuple[MetricName, str]],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    positions = range(len(metrics))
    for idx, (pipeline_name, pipeline_data) in enumerate(pipelines.items()):
        values = [pipeline_data["summary"].get(key, 0) for key, _ in metrics]
        offsets = [pos + idx * width for pos in positions]
        ax.bar(offsets, values, width=width, label=pipeline_name)
    ax.set_xticks([pos + width / 2 for pos in positions])
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success & Completion Metrics")
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "metric_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def _plot_error_breakdown(
    pipelines: Dict[PipelineName, Dict],
    output_dir: Path,
) -> Path:
    error_keys = ["validity", "prerequisite", "hallucination"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    positions = range(len(pipelines))
    bottoms = [0] * len(pipelines)
    for error in error_keys:
        values = [
            pipelines[pipeline]["summary"].get("error_breakdown", {}).get(error, 0)
            for pipeline in pipelines
        ]
        ax.bar(positions, values, width=width, bottom=bottoms, label=error.title())
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(pipelines.keys()))
    ax.set_ylabel("Fraction of errors")
    ax.set_title("Error Category Breakdown")
    ax.legend()
    fig.tight_layout()
    plot_path = output_dir / "error_breakdown.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def _collect_task_records(pipeline_runs: Iterable[Dict]) -> List[Dict]:
    records: List[Dict] = []
    for run in pipeline_runs:
        for task in run.get("tasks", []):
            records.append(task)
    return records


def _plot_reprompt_success_curve(
    pipelines: Dict[PipelineName, Dict], output_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, data in pipelines.items():
        records = _collect_task_records(data.get("runs", []))
        if not records:
            continue
        buckets: Dict[int, List[float]] = {}
        for rec in records:
            count = int(rec.get("reprompt_count", 0))
            buckets.setdefault(count, []).append(float(rec.get("success", 0)))
        xs = sorted(buckets.keys())
        ys = [sum(vals) / len(vals) for vals in (buckets[x] for x in xs)]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Reprompt count")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.set_title("How many reprompts until success?")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plot_path = output_dir / "reprompt_success_curve.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def _plot_planning_time(
    pipelines: Dict[PipelineName, Dict], output_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    widths = 0.6
    means = []
    labels = []
    for name, data in pipelines.items():
        records = _collect_task_records(data.get("runs", []))
        if not records:
            continue
        planning_times = [float(r.get("planning_time_sec", 0)) for r in records]
        if not planning_times:
            continue
        means.append(sum(planning_times) / len(planning_times))
        labels.append(name)

    positions = range(len(means))
    ax.bar(positions, means, width=widths)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average planning time (sec)")
    ax.set_title("Planning time per pipeline")
    fig.tight_layout()
    plot_path = output_dir / "planning_time.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def _write_metric_table(pipelines: Dict[PipelineName, Dict], output_dir: Path) -> Path:
    lines = [
        "pipeline,metric,value",
    ]
    for name, data in pipelines.items():
        for metric, value in data.get("summary", {}).items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    lines.append(f"{name},{metric}.{subkey},{subval}")
            else:
                lines.append(f"{name},{metric},{value}")
    csv_path = output_dir / "metric_summary.csv"
    csv_path.write_text("\n".join(lines))
    return csv_path


def build_plots(comparison_path: Path, output_dir: Path) -> Dict[str, Path]:
    data = _load_comparison(comparison_path)
    pipelines = data.get("pipelines", {})
    if not pipelines:
        raise ValueError("No pipeline data found in comparison file")

    _ensure_dir(output_dir)

    metrics = [
        ("task_success_rate", "Task Success"),
        ("partial_completion", "Partial Completion"),
        ("subgoal_grounding_success", "Grounding Success"),
        ("tamp_success_rate", "TAMP Success"),
    ]

    outputs = {
        "metric_bar": _plot_metric_bars(pipelines, metrics, output_dir),
        "errors": _plot_error_breakdown(pipelines, output_dir),
        "reprompt_curve": _plot_reprompt_success_curve(pipelines, output_dir),
        "planning_time": _plot_planning_time(pipelines, output_dir),
        "metric_csv": _write_metric_table(pipelines, output_dir),
    }
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=Path("./output/constraint_eval/comparison/pipeline_comparison.json"),
        help="Path to pipeline_comparison.json produced by evaluate_pipelines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/constraint_eval/plots"),
        help="Directory to write plots and CSV summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_plots(args.comparison_path, args.output_dir)
    print("Saved plots and summary:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
