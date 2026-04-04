# ALCS / ACS2 Library

## Overview

This repository contains the current ACS2-based implementation of the **Anticipatory Learning Classifier System** library available at:

[https://github.com/ounold/ALCS](https://github.com/ounold/ALCS)

The current implementation was developed in an **AI-assisted programming** workflow, with iterative human supervision, validation, and refinement of both the research code and the experimental tooling.

The codebase provides a unified entry point for running ACS2 experiments across:

- `cpu_single` for sequential CPU execution,
- `cpu_mp` for multiprocessing CPU execution,
- `gpu` for tensorized GPU execution.

The main executable is [`acs2.py`](acs2.py). It supports full single-backend runs as well as mixed backend schedules across the three ACS2 phases:

- `explore`
- `exploit1`
- `exploit2`

After each run, the program generates a dashboard in `reports/` summarizing learning progress, exploit performance, rule-population behavior, and selected structural metrics.

For a full parameter reference, see [parameter_guide.md](parameter_guide.md).

## Main Features

- Unified ACS2 runner through `acs2.py`
- Three execution backends: `cpu_single`, `cpu_mp`, `gpu`
- Mixed-mode execution through `UniversalRunner`
- Dashboard generation after each run
- Support for maze, multiplexer, binary-classification, and Gymnasium environments
- Batch maze benchmarking through `run_maze_benchmarks.py`
- Optional `no_subsumption` mode for ablation and backend-comparison studies
- Reproducible execution through an explicit `seed` parameter

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ounold/ALCS.git
cd ALCS
```

### 2. Create and activate a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch gymnasium numpy pandas matplotlib pyyaml
```

## Quick Start

### Full CPU single-core run

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --explore_mode cpu_single --exploit_mode cpu_single --device cpu
```

### Full multiprocessing CPU run

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --explore_mode cpu_mp --exploit_mode cpu_mp --device cpu
```

### Full GPU run

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --explore_mode gpu --exploit_mode gpu --device cuda
```

### Hybrid run

GPU in `explore`, CPU in both exploit phases:

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --explore_mode gpu --exploit_mode cpu_mp --device cuda
```

### Disable subsumption

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --explore_mode cpu_single --exploit_mode cpu_single --device cpu --no_subsumption true
```

## Using `acs2.py`

The main script:

- loads YAML defaults,
- applies CLI overrides,
- builds an experiment configuration,
- runs all three phases through `UniversalRunner`,
- computes final summary metrics,
- generates a dashboard in `reports/`.

### General command pattern

```powershell
.\.venv\Scripts\python.exe acs2.py --config <config.yaml> --explore_mode <cpu_single|cpu_mp|gpu> --exploit_mode <cpu_single|cpu_mp|gpu> --device <cpu|cuda|auto>
```

If `--exploit_mode` is omitted, `acs2.py` defaults to `cpu_single` for `exploit1` and `exploit2`.

## Environment Examples

## 1. Maze environments

Maze environments are resolved by name from `environment/acs2_mazes`.

Example: `Woods1`

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --environment_type grid_maze --environment_name Woods1 --explore_mode cpu_single --exploit_mode cpu_single --device cpu
```

Example: `MazeE3` with multiprocessing

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --environment_type grid_maze --environment_name MazeE3 --explore_mode cpu_mp --exploit_mode cpu_mp --device cpu --no_subsumption true
```

Example: full GPU maze run

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --environment_type grid_maze --environment_name Cassandra4x4 --explore_mode gpu --exploit_mode gpu --device cuda --no_subsumption true
```

## 2. Multiplexer environments

For multiplexer tasks, use `environment_type=multiplexer`.

Example:

```powershell
.\.venv\Scripts\python.exe acs2.py --environment_type multiplexer --environment_name mux_11 --address_bits 3 --sampling random --n_steps 1 --n_exp 10 --explore_mode cpu_mp --exploit_mode cpu_mp --device cpu
```

A 3-bit address multiplexer implies `3 + 2^3 = 11` input bits.

For multiplexer-like tasks, `n_steps=1` is usually the correct setting.

## 3. Binary-classification environments

Example: even parity

```powershell
.\.venv\Scripts\python.exe acs2.py --environment_type binary_classification --environment_name even_parity_6 --problem_kind even_parity --input_bits 6 --n_steps 1 --explore_mode cpu_single --exploit_mode cpu_single --device cpu
```

Example: carry

```powershell
.\.venv\Scripts\python.exe acs2.py --environment_type binary_classification --environment_name carry_8 --problem_kind carry --left_bits 4 --right_bits 4 --n_steps 1 --explore_mode cpu_mp --exploit_mode cpu_mp --device cpu
```

## 4. Gymnasium environments

Gymnasium environments use `environment_type=gymnasium`.

Example: FrozenLake

```powershell
.\.venv\Scripts\python.exe acs2.py --environment_type gymnasium --environment_name FrozenLake-v1 --env_id FrozenLake-v1 --is_slippery false --observation_encoding auto --n_exp 10 --n_steps 100 --explore_mode cpu_single --exploit_mode cpu_single --device cpu
```

Example: CartPole with binned observations

```powershell
.\.venv\Scripts\python.exe acs2.py --environment_type gymnasium --environment_name CartPole-v1 --env_id CartPole-v1 --observation_encoding binned --bins 8 8 8 8 --n_exp 10 --n_steps 200 --explore_mode cpu_single --exploit_mode cpu_single --device cpu
```

## Batch Benchmarking with `run_maze_benchmarks.py`

The helper script [`run_maze_benchmarks.py`](run_maze_benchmarks.py) runs the maze benchmark over:

- all mazes from `environment/acs2_mazes`, or a selected subset from YAML,
- all modes from YAML, or the default set:
  - `CPU Single`
  - `CPU MP`
  - `GPU`

It stores a CSV with:

- `total_time_s`
- `avg_exp_time_s`
- `std_exp_time_s`
- `exploit_avg_steps`
- `exploit_avg_steps_std`
- GPU timing breakdown columns

### Run the full benchmark from YAML

```powershell
.\.venv\Scripts\python.exe run_maze_benchmarks.py --config experiments/configs/batch_mazes.yaml
```

### Run the full benchmark and save to a specific file

```powershell
.\.venv\Scripts\python.exe run_maze_benchmarks.py --config experiments/configs/batch_mazes.yaml --output reports\maze_benchmarks_full.csv
```

### Force no-subsumption from CLI

```powershell
.\.venv\Scripts\python.exe run_maze_benchmarks.py --config experiments/configs/batch_mazes.yaml --no_subsumption --output reports\maze_benchmarks_no_subsumption.csv
```

### Restrict modes and mazes from YAML

`run_maze_benchmarks.py` also supports YAML keys:

- `mazes:`
- `modes:`

Example:

```yaml
mazes:
  - Woods1
  - MazeE3

modes:
  - cpu_single
  - cpu_mp
```

Then run:

```powershell
.\.venv\Scripts\python.exe run_maze_benchmarks.py --config experiments/configs/my_subset.yaml
```

## Dashboards

After every `acs2.py` run, a dashboard PNG is saved in `reports/`.

The dashboard includes:

- steps to goal over episodes,
- population size,
- knowledge and generalization,
- reward and quality,
- policy map,
- top rules,
- rule-origin distributions,
- creation-distribution summaries,
- final textual summary.

### Save raw dashboard data

```powershell
.\.venv\Scripts\python.exe acs2.py --config experiments/configs/batch_mazes.yaml --explore_mode cpu_single --exploit_mode cpu_single --device cpu --save_dashboard_data
```

### Load and re-render saved dashboard data

```powershell
.\.venv\Scripts\python.exe acs2.py --load_dashboard_data <timestamp> --plot_all_dashboards
```

You can also render only selected plots with:

- `--plot_steps`
- `--plot_population`
- `--plot_knowledge`
- `--plot_reward_quality`
- `--plot_policy_map`
- `--plot_top_rules`
- `--plot_origin_distribution`
- `--plot_origin_distribution_abs`
- `--plot_creation_dist <key>`

## Repository Structure

```text
ALCS/
|-- acs2.py
|-- run_maze_benchmarks.py
|-- experiments/
|   `-- configs/
|-- environment/
|   |-- acs2_mazes/
|   |-- registry.py
|   |-- runtime_cpu3.py
|   `-- runtime_gpu4.py
|-- reports/
|   `-- saved_states/
|-- src/
|   |-- universal_runner.py
|   |-- hybrid_utils.py
|   |-- configCPU3.py
|   |-- configGPU4.py
|   |-- experiment_runnerCPU3.py
|   |-- experiment_runnerGPU4.py
|   |-- visualizationCPU3.py
|   `-- visualizationGPU4.py
`-- parameter_guide.md
```

## Notes on Current Semantics

- `cpu_single` and `cpu_mp` use the same ACS2 learner; `cpu_mp` parallelizes independent experiments.
- `gpu` is a tensorized backend and is not behaviorally identical to the CPU implementations.
- `Exploit Avg. Steps` is computed from the `exploit2` slice of `stats_steps`.
- Maze names passed through `--environment_name` now resolve correctly to the real ACS2 maze definitions.

## Documentation

- Main parameter reference: [parameter_guide.md](parameter_guide.md)
- Benchmark helper: [run_maze_benchmarks.py](run_maze_benchmarks.py)
- Main entry point: [acs2.py](acs2.py)

## Citation

If you use this library, please cite the repository:

```text
Olgierd Unold, "ALCS / ACS2 Library", GitHub repository, 2026.
URL: https://github.com/ounold/ALCS
```

## Contact

Please feel free to contact me at: [olgierd.unold@pwr.edu.pl](mailto:olgierd.unold@pwr.edu.pl)

## License

This project is licensed under the MIT License. 
