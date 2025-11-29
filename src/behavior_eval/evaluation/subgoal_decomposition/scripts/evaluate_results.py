import csv
import json
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple
from multiprocessing import Process, Manager, Queue

import behavior_eval
import fire

from behavior_eval.evaluation.subgoal_decomposition.subgoal_plan import (
    SubgoalPlanHalfJson,
)
from behavior_eval.evaluation.subgoal_decomposition.subgoal_sim_utils import (
    EvalStatistics,
    evaluate_task,
    get_all_raw_task_goal,
    get_all_task_list,
)
from behavior_eval.evaluation.subgoal_decomposition.subgoal_eval_utils import (
    extract_model_names,
    goal_eval_stats,
    traj_eval_stats,
)

def simulate_llm_response(demo_name, lock, llm_plan_path, eval_stat_path):
    report = evaluate_task(demo_name, llm_plan_path)
    goal_info = report[-1]
    with lock:
        eval_statistics = EvalStatistics(get_all_task_list(), eval_stat_path)
        if report[0] != 'Correct':
            eval_statistics.update_eval_rst_dict(demo_name, False, str(report[:-1]), goal_info)
        else:
            eval_statistics.update_eval_rst_dict(demo_name, True, str(report[:-1]), goal_info)
        eval_statistics.save_eval_rst_dict()

def worker_task(queue, lock, eval_stat_path):
    while True:
        task = queue.get()
        if task is None:
            break
        demo_name, llm_plan_path = task
        simulate_llm_response(demo_name, lock, llm_plan_path, eval_stat_path)


def simulate_one_llm(llm_response_path, llm_name: str, worker_num: int=1, result_dir: str='./results'):
    get_all_raw_task_goal()
    manager = Manager()
    lock = manager.Lock()

    # llm_name = os.path.basename(llm_response_path).split('_')[0]
    eval_stat_path = os.path.join(result_dir, 'log', f'{llm_name}.json')
    os.makedirs(os.path.dirname(eval_stat_path), exist_ok=True)

    task_list = get_all_task_list()
    cur_eval_stats = EvalStatistics(task_list, eval_stat_path)
    real_task_list = [task_name for task_name in task_list if not cur_eval_stats.check_evaluated_task(task_name)]

    if worker_num > 1:
        worker_num = min(worker_num, len(real_task_list))
        task_queue = Queue()
        workers = []
        for i in range(worker_num):
            worker = Process(target=worker_task, args=(task_queue, lock, eval_stat_path))
            worker.start()
            workers.append(worker)
        
        for demo_name in real_task_list:
            task_queue.put((demo_name, llm_response_path))
        for i in range(worker_num):
            task_queue.put(None)
        for worker in workers:
            worker.join()
    else:
        for demo_name in real_task_list:
            simulate_llm_response(demo_name, lock, llm_response_path, eval_stat_path)
    print(f'Results saved to {eval_stat_path}')
    summary = {
        "trajectory_evaluation": {},
        "goal_evaluation": {}
    }
    traj_stats = traj_eval_stats(eval_stat_path)
    goal_stats = goal_eval_stats(eval_stat_path)
    summary['trajectory_evaluation'] = traj_stats
    summary['goal_evaluation'] = goal_stats

    summary_path = os.path.join(result_dir, 'summary', f'{llm_name}.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    return summary

def evaluate_results(llm_response_dir, worker_num: int=1, result_dir: str='./results'):
    os.makedirs(result_dir, exist_ok=True)
    available_model_names = extract_model_names(llm_response_dir)
    if not available_model_names:
        print('No model found in the directory')
        return
    for model_name in available_model_names:
        model_path = os.path.join(llm_response_dir, f'{model_name}_outputs.json')
        assert os.path.exists(model_path), f'{model_path} not found in the directory'
        simulate_one_llm(model_path, model_name, worker_num, result_dir)


def _extract_goal_progress(goal_info: Optional[Dict[str, Any]]) -> Tuple[float, bool]:
    """Return partial completion ratio and whether all subgoals were grounded."""
    if not isinstance(goal_info, dict):
        return 0.0, False
    subgoal_success = goal_info.get("subgoal_success") or []
    total = len(subgoal_success)
    successes = sum(1 for s in subgoal_success if s)
    partial_completion = successes / total if total > 0 else 0.0
    return partial_completion, bool(total > 0 and successes == total)


def _classify_error(label: Optional[str]) -> Optional[str]:
    if label is None or label == "Correct":
        return None
    if label in {"NotParseable"}:
        return "validity"
    if label == "Hallucination":
        return "hallucination"
    if label in {"Runtime", "GoalUnreachable"}:
        return "prerequisite"
    return "validity"


def _aggregate_run_metrics(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(task_results) or 1
    error_counts = {"validity": 0, "prerequisite": 0, "hallucination": 0}
    for result in task_results:
        category = result["error_category"]
        if category:
            error_counts[category] += 1

    def _avg(key: str) -> float:
        return round(sum(r[key] for r in task_results) / total, 4)

    return {
        "task_success_rate": _avg("success"),
        "partial_completion": _avg("partial_completion"),
        "subgoal_grounding_success": _avg("subgoal_grounding_success"),
        "tamp_success_rate": _avg("tamp_success"),
        "error_breakdown": {k: round(v / total, 4) for k, v in error_counts.items()},
        "avg_reprompt_count": _avg("reprompt_count"),
        "avg_planning_time_sec": _avg("planning_time_sec"),
    }


def _build_task_filter(llm_response_path: str) -> Dict[str, str]:
    """Map task identifiers to raw llm outputs for fast lookup."""
    with open(llm_response_path, "r") as f:
        plans = json.load(f)
    task_map: Dict[str, str] = {}
    if isinstance(plans, list):
        for plan in plans:
            identifier = plan.get("identifier") or plan.get("task_name")
            if identifier:
                task_map[identifier] = plan.get("llm_output", "")
    elif isinstance(plans, dict):
        for identifier, plan in plans.items():
            if isinstance(plan, dict):
                task_map[identifier] = plan.get("output") or plan.get("llm_output", "")
    return task_map


def _estimate_plan_length(raw_llm_output: str, seed: int) -> int:
    """Estimate subgoal length using the HalfJson parsing pipeline."""
    if not raw_llm_output:
        return 0
    json_obj = SubgoalPlanHalfJson.extract_json_obj(raw_llm_output)
    if json_obj is None:
        return 0
    rng_state = random.getstate()
    random.seed(seed)
    try:
        subgoals = SubgoalPlanHalfJson.preprocess_raw_plan_obj(json_obj)
    finally:
        random.setstate(rng_state)
    return len(subgoals)


def _select_tasks_by_length(
    llm_response_path: str,
    min_steps: int,
    max_steps: int,
    sample_size: Optional[int],
    seed: int,
) -> List[str]:
    task_map = _build_task_filter(llm_response_path)
    available_tasks = set(get_all_task_list())
    rng = random.Random(seed)
    candidates = []
    for identifier, llm_output in task_map.items():
        if identifier not in available_tasks:
            continue
        length = _estimate_plan_length(llm_output, seed)
        if min_steps <= length <= max_steps:
            candidates.append(identifier)
    rng.shuffle(candidates)
    if sample_size is not None:
        candidates = candidates[:sample_size]
    return candidates


def _run_pipeline_once(
    tasks: Sequence[str],
    llm_response_path: str,
    constraint_mode: Optional[str],
    run_seed: int,
) -> Dict[str, Any]:
    task_results: List[Dict[str, Any]] = []
    for task in tasks:
        start = time.perf_counter()
        report, metadata = evaluate_task(
            task,
            llm_response_path,
            constraint_mode=constraint_mode,
            return_metadata=True,
        )
        elapsed = time.perf_counter() - start
        label = report[0] if isinstance(report, tuple) and len(report) > 0 else None
        goal_info = report[-1] if isinstance(report, tuple) and len(report) > 0 else None
        partial_completion, grounding_success = _extract_goal_progress(goal_info)
        tamp_success = bool(report[1]) if isinstance(report, tuple) and len(report) > 1 else False
        reprompt_count = int((metadata or {}).get("reprompt_count", 0))
        task_results.append(
            {
                "task_name": task,
                "success": float(label == "Correct"),
                "partial_completion": partial_completion,
                "subgoal_grounding_success": float(grounding_success),
                "tamp_success": float(tamp_success),
                "error_category": _classify_error(label),
                "reprompt_count": reprompt_count,
                "planning_time_sec": elapsed,
            }
        )
    return {
        "run_seed": run_seed,
        "tasks": task_results,
        "aggregate": _aggregate_run_metrics(task_results),
    }


def _summarize_pipeline_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {}
    keys = [
        "task_success_rate",
        "partial_completion",
        "subgoal_grounding_success",
        "tamp_success_rate",
        "avg_reprompt_count",
        "avg_planning_time_sec",
    ]
    summary: Dict[str, Any] = {}
    for key in keys:
        summary[key] = round(mean(run["aggregate"][key] for run in runs), 4)
    # Average nested error breakdown
    breakdown_keys = runs[0]["aggregate"]["error_breakdown"].keys()
    summary["error_breakdown"] = {
        k: round(mean(run["aggregate"]["error_breakdown"][k] for run in runs), 4)
        for k in breakdown_keys
    }
    return summary


def _write_summary_outputs(
    output_dir: Path,
    pipelines: Dict[str, Dict[str, Any]],
    sampled_tasks: Sequence[str],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pipeline_comparison.json"
    with open(json_path, "w") as f:
        json.dump({"pipelines": pipelines, "sampled_tasks": list(sampled_tasks)}, f, indent=2)

    csv_path = output_dir / "pipeline_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "metric", "value"])
        for name, data in pipelines.items():
            summary = data["summary"]
            for key, value in summary.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        writer.writerow([name, f"{key}.{subkey}", subval])
                else:
                    writer.writerow([name, key, value])

    table_path = output_dir / "pipeline_summary.md"
    metrics_headers = [
        "task_success_rate",
        "partial_completion",
        "subgoal_grounding_success",
        "tamp_success_rate",
        "avg_reprompt_count",
        "avg_planning_time_sec",
    ]
    header_line = "| pipeline | " + " | ".join(metrics_headers) + " |"
    separator_line = "|---" * (len(metrics_headers) + 1) + "|"
    with open(table_path, "w") as f:
        f.write(header_line + "\n")
        f.write(separator_line + "\n")
        for name, data in pipelines.items():
            summary = data["summary"]
            row = [name] + [str(summary.get(metric, "")) for metric in metrics_headers]
            f.write("| " + " | ".join(row) + " |\n")

    plot_path = output_dir / "pipeline_summary.png"
    try:
        import matplotlib.pyplot as plt

        metrics = [
            ("task_success_rate", "Task Success"),
            ("partial_completion", "Partial Completion"),
            ("subgoal_grounding_success", "Subgoal Grounding"),
            ("tamp_success_rate", "TAMP Success"),
        ]
        pipelines_list = list(pipelines.keys())
        x = range(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        for idx, pipeline_name in enumerate(pipelines_list):
            values = [pipelines[pipeline_name]["summary"].get(m[0], 0) for m in metrics]
            offsets = [pos + idx * width for pos in x]
            ax.bar(offsets, values, width=width, label=pipeline_name)
        ax.set_xticks([pos + width / 2 for pos in x])
        ax.set_xticklabels([m[1] for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title("Baseline vs Constraint-Augmented")
        ax.legend()
        fig.tight_layout()
        plt.savefig(plot_path)
    except Exception:
        plot_path = None
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "table": str(table_path),
        "plot": str(plot_path) if plot_path else None,
    }


def evaluate_pipelines(
    llm_response_path: str,
    result_dir: str = "./results",
    runs: int = 3,
    min_steps: int = 10,
    max_steps: int = 30,
    constraint_mode: str = "affordance",
    sample_size: Optional[int] = None,
    seed: int = 0,
):
    """Run baseline and constraint-augmented pipelines on the same tasks.

    The function samples tasks with subgoal plans between ``min_steps`` and
    ``max_steps`` (inclusive), then evaluates both the baseline subgoal
    pipeline and a constraint-augmented variant multiple times to capture
    robust statistics.
    """

    comparison_dir = Path(result_dir) / "comparison"
    sampled_tasks = _select_tasks_by_length(
        llm_response_path, min_steps=min_steps, max_steps=max_steps, sample_size=sample_size, seed=seed
    )
    if not sampled_tasks:
        print("No tasks matched the requested step range.")
        return None

    pipelines: Dict[str, Dict[str, Any]] = {}
    pipeline_configs = {
        "baseline": None,
        "constraint_augmented": constraint_mode,
    }

    for name, constraint in pipeline_configs.items():
        runs_data = []
        for i in range(runs):
            run_seed = seed + i
            runs_data.append(
                _run_pipeline_once(
                    sampled_tasks,
                    llm_response_path,
                    constraint_mode=constraint,
                    run_seed=run_seed,
                )
            )
        pipelines[name] = {
            "runs": runs_data,
            "summary": _summarize_pipeline_runs(runs_data),
        }

    outputs = _write_summary_outputs(comparison_dir, pipelines, sampled_tasks)
    print(f"Saved pipeline comparison to {outputs}")
    return outputs

if __name__ == '__main__':
    fire.Fire(
        {
            "evaluate_results": evaluate_results,
            "evaluate_pipelines": evaluate_pipelines,
        }
    )
