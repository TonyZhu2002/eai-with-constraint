# Embodied Agent Interface 中文指南

本仓库提供 **Embodied Agent Interface (EAI)** 评测框架，统一支持 VirtualHome 与 BEHAVIOR 两个具身决策数据集，并覆盖目标理解、子目标分解、动作序列、状态转移建模四大模块。本文档用中文说明如何克隆项目、配置运行环境、配置 LLM/VLM API，以及详细的使用与结果解析方式，并新增一键复现实验脚本。

## 克隆仓库
```bash
git clone https://github.com/embodied-agent-interface/embodied-agent-interface.git
cd embodied-agent-interface
```

## 环境配置
1. **创建 Conda 环境（推荐）**
   ```bash
   conda create -n eai-eval python=3.8 -y
   conda activate eai-eval
   ```

2. **安装依赖**
   - 从 PyPI 安装：
     ```bash
     pip install eai-eval
     ```
   - 或从源码安装（便于本地修改）：
     ```bash
     pip install -e .
     ```

3. **可选依赖**
   - **PDDL planner**：若需要评测 transition_modeling，建议先运行 `python examples/pddl_tester.py` 确认安装成功。
   - **iGibson**：若使用 `behavior_eval`，执行 `python -m behavior_eval.utils.install_igibson_utils` 并按提示下载资源。

## 配置 LLM/VLM API
- 官方脚本默认使用 OpenAI 接口，请在运行前设置环境变量：
  ```bash
  export OPENAI_API_KEY="your_key"
  ```
- 若需要切换到自定义模型（包括 VLM），可在生成的提示文件上自行调用其它推理服务，只需将推理结果按原格式写回指定的 `llm_response_path`（见下文参数）。

## 基础命令行参数
核心入口为 `eai-eval`，重要参数如下：
- `--mode {generate_prompts, evaluate_results}`：切换模式。
  - `generate_prompts`：生成模型输入提示并保存在 `output/<dataset>/generate_prompts/<eval_type>/`。
  - `evaluate_results`：读取 `llm_response_path` 下的模型输出，生成评测结果。
- `--eval-type {action_sequencing, transition_modeling, goal_interpretation, subgoal_decomposition}`：选择评测模块。
- `--dataset {virtualhome, behavior}`：选择数据集。
- `--llm-response-path <path>`：自定义模型输出存放路径（默认使用内置 `helm_output`）。
- `--output-dir <path>`：评测结果输出目录（默认 `./output`）。
- `--num-workers <int>`：并行进程数。
- `--all`：自动遍历未指定的 `mode`、`eval-type`、`dataset` 组合，批量运行。

## 使用流程示例
1. **生成提示（切换到提示模式）**
   ```bash
   eai-eval --dataset virtualhome --eval-type action_sequencing --mode generate_prompts --output-dir ./output
   ```
   生成的 JSON/文本提示位于 `output/virtualhome/generate_prompts/action_sequencing/`，可直接送入任意 LLM/VLM。

2. **运行评测脚本（结果评测模式）**
   - 若使用官方 HELM 推理结果，先下载：
     ```bash
     python -m eai_eval.utils.download_utils
     ```
   - 评测某个模块：
     ```bash
     eai-eval --dataset behavior --eval-type transition_modeling --mode evaluate_results --output-dir ./output
     ```
   - 一次性评测所有模块：
     ```bash
     eai-eval --mode evaluate_results --all --output-dir ./output/full_run
     ```

3. **解析结果**
   - 每个模型的结果保存在 `output/<dataset>/evaluate_results/<eval_type>/<model_name>/`。
   - 关键文件：
     - `summary.json`：包含成功率、语法错误率、缺步/顺序/可供性等细分指标。
     - `error_info.json`（部分任务）：列出具体错误编号与统计，便于对症调试。
   - CLI 运行完会在终端提示结果目录，可直接用 `jq` 或任意 JSON 查看工具解析。

## 复现实验的一键脚本
- 运行脚本会自动下载官方 HELM 输出，并在全部数据集与评测类型上执行结果评测：
  ```bash
  bash scripts/run_all_evaluations.sh ./output/benchmark_run
  ```
- 你可以修改第一个参数更改输出目录；如需用自定义模型结果，只需在运行前将结果放入 `helm_output/<dataset>/<eval_type>/` 的同名 JSON 结构中，再执行脚本即可复现对比实验。

## 小贴士
- 首次运行会在 `output/` 下创建分层目录；重复运行会覆盖同名结果。
- 如需自定义日志，可在 `virtualhome_eval.log_config.setup_logging` 中调整级别；脚本结束后，终端会打印最终结果路径以便定位。
