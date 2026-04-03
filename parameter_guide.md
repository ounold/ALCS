# CPU and GPU Parameter Guide

## Purpose

This guide provides a comprehensive reference for selecting and configuring the execution modes in the ACS2 project. It covers all command-line parameters for the unified entry point:
- `acs2.py` (Unified CPU/GPU Framework)

## Execution Mode vs. Hardware Device

It is important to distinguish between **how** the experiments are distributed (Mode) and **where** the actual math is performed (Device).

### 1. `explore_mode` / `exploit_mode` (The Algorithm Backend)
These flags control the **architectural backend**. They decide which strategy manages the experiment phases:

*   **`gpu`**: Uses **Batched Execution**. It takes all `n_exp` experiments and packs them into huge tensors. All experiments are updated simultaneously in a single loop. Highly efficient for large populations and simple logic.
*   **`cpu_mp`**: Uses **Parallel Execution**. It starts multiple independent Python processes (one per CPU core). Each process runs one experiment at a time using standard object-oriented logic. Best for complex logic that is hard to vectorize.
*   **`cpu_single`**: Uses the same CPU object-oriented logic in a single process. Best for debugging and deterministic step-by-step inspection.
*   *Note:* `acs2.py` can now run fully on one backend or use a hybrid handoff between backends.

### 2. `device` (The Physical Hardware)
This flag is specific to the **GPU algorithm mode**. Once you choose the `gpu` backend, `device` tells PyTorch which physical hardware to use for those tensor calculations:

*   **`cuda`**: Tensors are sent to your NVIDIA graphics card. This provides maximum speed.
*   **`cpu`**: Tensors stay in RAM and are processed by your CPU's instruction sets (AVX/SSE). 
    *   *Note:* `mode: gpu` with `device: cpu` is often slower than `mode: cpu_mp` because you get the overhead of tensors without the benefit of multi-core process parallelization.
*   **`auto`**: Automatically detects if a CUDA-capable GPU is available.

### Summary Selection Matrix

| Backend Mode | Device Setting | Resulting Behavior |
| :--- | :--- | :--- |
| **`gpu`** | `cuda` | Tensor-batched math on Graphics Card (Fastest for Discovery). |
| **`gpu`** | `cpu` | Tensor-batched math on CPU (Testing GPU logic without a card). |
| **`cpu_mp`** | (Ignored) | Multi-core parallel processes (Fastest for CPU-based logic). |
| **`cpu_single`**| (Ignored) | One process, one core (Best for debugging). |

### Canonical Execution Patterns

| Command Pattern | Meaning |
| :--- | :--- |
| `--explore_mode cpu_single --exploit_mode cpu_single` | Full single-core CPU run. |
| `--explore_mode cpu_mp --exploit_mode cpu_mp` | Full multiprocessing CPU run. |
| `--explore_mode gpu --exploit_mode gpu` | Full GPU run across all phases. |
| `--explore_mode gpu --exploit_mode cpu_mp` | Hybrid run: GPU explore, CPU multiprocessing exploit. |
| `--explore_mode <mode>` | If `--exploit_mode` is omitted, `exploit1` and `exploit2` default to `cpu_single`. |

## Project Directory Structure

For the system to function correctly, files and results must be organized as follows:

```text
acs2_project/
├── acs2.py                 # Unified entry point
├── experiments/
│   └── configs/            # YAML configuration files (.yaml)
├── reports/                # Generated PNG dashboards
│   └── saved_states/       # Raw data (CSV, NPY, Metadata YAML)
├── src/                    # Core logic
│   ├── hybrid_utils.py     # Merging and translation logic
│   ├── universal_runner.py # Phase dispatcher and handoff
│   ├── models/             # ACS2 Agent implementations (CPU/GPU)
│   ├── config*.py          # Configuration models
│   ├── experiment_runner*.py# Hardware-specific loop runners
│   ├── data_handler*.py    # Saving and Loading mechanisms
│   └── visualization*.py   # Plotting and dashboard logic
├── environment/            # Environment definitions (Mazes, Multiplexers)
└── tests/                  # Verification tests
```

## Detailed Command-line Parameters

### Built-in Default Values

If not specified in a YAML config or via CLI, the following defaults are used:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `n_exp` | `5` | Number of averaged experiment runs. |
| `n_steps` | `100` | Max steps per episode. |
| `beta` | `0.05` | Global learning rate. |
| `gamma` | `0.95` | Discount factor. |
| `theta_i` | `0.1` | Inadequacy threshold. |
| `theta_r` | `0.9` | Reliability threshold. |
| `epsilon` | `0.1` | Global exploration rate. |
| `u_max` | `1` | Max attributes in covering. |
| `theta_ga` | `100` | GA invocation interval. |
| `mu` | `0.3` | Mutation probability. |
| `chi` | `0.8` | Crossover probability. |
| `theta_as` | `50` | Action set size limit. |
| `theta_exp` | `20` | Experience for subsumption. |
| `no_subsumption` | `false` | Disable subsumption in CPU and GPU backends. |
| `alp_mark_only_incorrect` | `true` | Restricted marking mode. |
| `metric_calculation_frequency` | `1` | Compute metrics every N episodes. |
| `max_population` | `1024` | GPU-only population limit. |
| `device` | `auto` | Auto-select CUDA if available. |

**Phase Defaults (Total 2500 Episodes):**

| Phase | Episodes | Epsilon | Beta | ALP | GA | Decay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **explore** | 1500 | 0.8 | 0.05 | ON | ON | OFF |
| **exploit1** | 500 | 0.2 | 0.05 | ON | OFF | OFF |
| **exploit2** | 500 | 0.0 | 0.05 | OFF | OFF | OFF |

### Execution Settings
- `--explore_mode <cpu_single|cpu_mp|gpu>`: Backend used for the `explore` phase.
- `--exploit_mode <cpu_single|cpu_mp|gpu>`: Backend used for `exploit1` and `exploit2`. Default: `cpu_single`.
- `--device <auto|cpu|cuda>`: Physical device used by the GPU backend.

### General Experiment Settings
- `--config <path>`: Path to YAML config (e.g., `experiments/configs/mux_11.yaml`).
- `--n_exp <int>`: Number of averaged experiment runs.
- `--n_steps <int>`: Max steps per episode. (Set to `1` for Multiplexer/Parity).
- `--beta <float>`: Learning rate (0.0 - 1.0).
- `--gamma <float>`: Discount factor (0.0 - 1.0).
- `--theta_i <float>`: Inadequacy threshold for deletion.
- `--theta_r <float>`: Reliability threshold for metrics/knowledge.
- `--epsilon <float>`: General exploration probability.
- `--u_max <int>`: Max attributes in covering (smaller = more general).
- `--steps_to_goal_threshold <int>`: Optional threshold used by some evaluation/selection workflows.
- `--metric_calculation_frequency <int>`: Metric computation interval.
- `--alp_mark_only_incorrect <bool>`: **ALP Marking Strategy**. This flag is directly part of the Anticipatory Learning Process (ALP) and controls how environmental features are recorded in a classifier's "Mark" ($M$):
    - **`true` (Restricted Marking):** Marks are added **ONLY** to classifiers that made an **incorrect** prediction. The agent only seeks to specialize a rule when it is proven wrong. This strongly favors **generalization** and keeps the population compact.
    - **`false` (Full Action Set Marking):** Marks are added to **EVERY** classifier in the action set on every step, regardless of accuracy. This leads to **rapid specialization** and captures environmental nuances faster, but often results in a significantly larger, more specific population (linear growth).

    | Feature | Restricted (`true`) | Full Action Set (`false`) |
    | :--- | :--- | :--- |
    | **Population Size** | Compact (Consolidates into rules) | Large (Captures specific states) |
    | **Generalization** | High (Logical focus) | Low (Contextual focus) |
    | **Learning Speed** | Slower initial discovery | Rapid initial specialization |
    | **Recommended For** | Standard logic tasks (Mux, Parity) | Complex/Noisy environments |
- `--no_subsumption [true|false]`: Disable classifier subsumption in the active backend(s). This applies to CPU, GPU, and hybrid runs. The default is `false`, meaning standard subsumption is enabled.

### Genetic Algorithm (GA) Settings
- `--theta_ga <int>`: Episodes between GA triggers.
- `--mu <float>`: Mutation probability.
- `--chi <float>`: Crossover probability.
- `--theta_as <int>`: Action set size limit (triggers deletion).
- `--theta_exp <int>`: Experience required for subsumption.
- `--no_subsumption <bool>`: Turns subsumption off entirely while keeping the rest of ALP and GA active.
- `--max_population <int>`: GPU-only hard cap on classifier population tensors.

### Environment Selection
- `--environment_type <str>`: High-level environment family.
- `--environment_name <str>`: Specific registered environment name.

### Environment Parameters
These are environment-dependent. Only a subset is meaningful for a given environment:

- `--rows <int>`: Grid height for maze-like environments.
- `--cols <int>`: Grid width for maze-like environments.
- `--start_pos <row col>`: Start position as two integers.
- `--goal_pos <row col>`: Goal position as two integers.
- `--obstacles <r1 c1 r2 c2 ...>`: Flat list of obstacle coordinates; values are paired into `(row, col)` entries.
- `--reset_to_random_start <bool>`: Randomize the starting position between episodes when supported.
- `--address_bits <int>`: Number of address bits for multiplexer tasks.
- `--problem_kind <str>`: Problem type for boolean environments such as parity/carry.
- `--input_bits <int>`: Total input width for boolean environments.
- `--left_bits <int>`: Left operand size for split-input tasks.
- `--right_bits <int>`: Right operand size for split-input tasks.
- `--sampling <str>`: Sampling mode for generated environments.
- `--env_id <str>`: Gymnasium environment identifier when using Gym-based environments.
- `--observation_encoding <str>`: Observation preprocessing mode, typically `auto` unless you need a specific encoding.
- `--bins <int ...>`: Optional discretization bins for continuous observations.
- `--is_slippery <bool>`: FrozenLake-style transition stochasticity flag.

### Phase-Specific Settings (`explore`, `exploit1`, `exploit2`)
- `--<phase>_episodes <int>`: Phase length.
- `--<phase>_epsilon <float>`: Phase exploration rate.
- `--<phase>_beta <float>`: Phase learning rate.
- `--<phase>_alp <bool>`: Toggle ALP for this phase.
- `--<phase>_ga <bool>`: Toggle GA for this phase.
- `--<phase>_decay <bool>`: Toggle epsilon decay.

### Dashboard & Plotting
- `--save_dashboard_data`: Export raw data to `reports/saved_states/`.
- `--load_dashboard_data <timestamp>`: Load previously saved data and generate plots without re-running training.
- `--plot_all_dashboards`: Generate the full report (ends in `_all.png`).
- `--plot_steps`: Show Steps-to-Goal chart.
- `--plot_population`: Show Micro/Macro population chart.
- `--plot_knowledge`: Show Knowledge and Generalization chart.
- `--plot_reward_quality`: Show Reward and Quality chart.
- `--plot_policy_map`: Show policy map (maze-like environments only).
- `--plot_top_rules`: Show the top-rules table.
- `--plot_origin_distribution`: Show relative origin distribution over time.
- `--plot_origin_distribution_abs`: Show Absolute Origin counts.
- `--plot_creation_dist <key>`: Show birth episodes for a rule type.

**Plotting behavior:**
- During a normal experiment run, if you do **not** provide any `--plot_*` toggle, `acs2.py` generates the full dashboard by default.
- During a `--load_dashboard_data` run, you can request one or more specific plots using the `--plot_*` flags.

## Dashboard Metric Definitions

The final dashboard displays several key performance indicators (KPIs):

- **Steps to Goal:** The number of steps the agent took to reach the goal state in an episode. Lower is better.
- **Knowledge (%):** The percentage of the environment's state-action space for which the agent has a **reliable** and **correct** predicting classifier.
- **Generalization (%):** The ratio of wildcards (`#`) to total attributes in the population. Higher indicates a more compact, generalized knowledge base.
- **Micro-population:** The total number of classifiers in the population, including their numerosities (copies).
- **Macro-population:** The number of unique classifiers in the population.
- **Reliable (Rel):** The average calculated exclusively for the subset of rules meeting the reliability threshold ($q > \theta_r$). 
- **Reward (R):** The average internal reward prediction of the classifiers.
- **Quality (Q):** The average accuracy/reliability of the classifiers' predictions.


## Generating Separate Charts

By default, `acs2.py` generates a comprehensive dashboard (`_all.png`) when no specific plot toggle is provided. To generate individual charts for specific metrics (for example, for inclusion in a paper), use `--load_dashboard_data` with one or more plot toggles:

1. **Find the timestamp** of your run in `reports/saved_states/` (e.g., `20260329_123456`).
2. **Run the command** with the desired toggles:

```powershell
# Example: Generate only the Knowledge and Population charts
.\.venv\Scripts\python.exe acs2.py --load_dashboard_data 20260329_123456 --plot_knowledge --plot_population
```

**Available Toggles:**
- `--plot_steps`
- `--plot_population`
- `--plot_knowledge`
- `--plot_reward_quality`
- `--plot_policy_map` (Mazes only)
- `--plot_top_rules`
- `--plot_origin_distribution`
- `--plot_origin_distribution_abs`
- `--plot_creation_dist <key>` (Keys: `alp_expected`, `covering`, `ga`, `alp_unexpected`, `alp_covering`)

## Practical Recommendations

1. **Multiplexer Runs:** Always use `n_steps: 1`.
2. **Linear Population Growth:** If the population doesn't plateau, consider enabling Action Set Subsumption (standard ACS2).
3. **Knowledge Reporting:** Ensure your classifiers reach `theta_r` (0.9) to be counted as "knowing" the environment.
4. **Default Workflow:** If you set only `--explore_mode`, the two exploit phases run in `cpu_single`.
5. **Hybrid Workflow:** Use `acs2.py --explore_mode gpu --exploit_mode cpu_mp` when you want fast tensorized discovery and CPU-based exploit evaluation.
6. **Full GPU Workflow:** Use `acs2.py --explore_mode gpu --exploit_mode gpu --device cuda` when you want every phase to stay on the GPU backend.
7. **No-Subsumption Ablation:** Use `acs2.py --no_subsumption ...` when you want a direct CPU/GPU comparison without subsumption overhead.

## Maze Benchmark Helper

The batch benchmark script supports the same subsumption toggle:

- `python run_maze_benchmarks.py`
- `python run_maze_benchmarks.py --no_subsumption`

When `--no_subsumption` is used, the output CSV includes a `no_subsumption` column so the benchmark setting is recorded explicitly.
