# ACS2 Unified CPU/GPU Framework

## Overview

This project provides a highly optimized, unified framework for the **Anticipatory Classifier System 2 (ACS2)**. It bridges the gap between traditional object-oriented CPU implementations and high-performance tensor-batched GPU execution. 

The framework is designed for research and experimentation in Reinforcement Learning, offering seamless transitions between hardware backends and comprehensive visualization dashboards.

## Key Features

*   **Unified Entry Point:** A single `acs2.py` script to manage all experiment types.
*   **Hybrid Execution:** Run fast discovery phases on the **GPU** and switch to parallel **CPU** processes for detailed exploitation analysis.
*   **Full GPU Execution:** Run `explore`, `exploit1`, and `exploit2` entirely on the GPU backend when you want end-to-end tensorized execution.
*   **Tensorized GPU Backend:** Process thousands of experiments simultaneously using PyTorch tensors.
*   **Parallel CPU Backend:** Utilize all available CPU cores for complex logic and object-level analysis.
*   **Single-Core Exploit Default:** `acs2.py` now defaults the exploit phases to `cpu_single` unless you explicitly choose another backend.
*   **Optional Subsumption Disable:** Use `--no_subsumption` to disable subsumption in either CPU or GPU backends for diagnostics and performance studies.
*   **Automated Knowledge Handoff:** Seamlessly transfer rule populations from GPU memory to CPU structures.
*   **Comprehensive Dashboards:** Generate detailed reports with metrics for knowledge, generalization, population dynamics, and rule origin.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/acs2-project.git
    cd acs2-project
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows:
    .\.venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch gymnasium numpy pandas matplotlib pyyaml
    ```

## Quick Start

### 1. Standard CPU Run (Multi-core)
Run a multiplexer experiment using all available CPU cores:
```bash
python acs2.py --config experiments/configs/mux_11.yaml --explore_mode cpu_mp --exploit_mode cpu_mp
```

### 2. Fast GPU Exploration
Utilize your NVIDIA GPU for massive parallel discovery:
```bash
python acs2.py --config experiments/configs/mux_11.yaml --explore_mode gpu --device cuda
```

### 3. Full GPU Run
Keep every phase on the GPU backend:
```bash
python acs2.py --config experiments/configs/mux_11.yaml --explore_mode gpu --exploit_mode gpu --device cuda
```

### 4. Hybrid GPU -> CPU Handoff
Discovery on GPU, then switch to CPU for stable performance measurement:
```bash
python acs2.py --config experiments/configs/mux_11.yaml --explore_mode gpu
```

### 5. Run Without Subsumption
Disable subsumption in all active backends:
```bash
python acs2.py --config experiments/configs/mux_11.yaml --no_subsumption
```

## Directory Structure

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
│   ├── config*.py          # Configuration models (CPU/GPU)
│   ├── experiment_runner*.py# Hardware-specific loop runners
│   ├── data_handler*.py    # Saving and Loading mechanisms
│   ├── metrics*.py         # Metric calculation logic
│   └── visualization*.py   # Plotting and dashboard logic
├── environment/            # Environment definitions (Mazes, Multiplexers)
└── tests/                  # Verification tests
```

## Configuration

The system uses YAML files for primary configuration. Place your configurations in `experiments/configs/`. 

### Execution Backends
`acs2.py` now supports:
*   Full CPU single-core runs: `--explore_mode cpu_single --exploit_mode cpu_single`
*   Full CPU multiprocessing runs: `--explore_mode cpu_mp --exploit_mode cpu_mp`
*   Full GPU runs: `--explore_mode gpu --exploit_mode gpu`
*   Hybrid runs: `--explore_mode gpu --exploit_mode cpu_mp`
*   Default behavior when `--exploit_mode` is omitted: `cpu_single`
*   Optional subsumption disable across CPU and GPU runs: `--no_subsumption`

### ALP Marking Strategy
A critical scientific parameter is `--alp_mark_only_incorrect`:
*   **`true` (Restricted):** Favors **generalization** and compact populations.
*   **`false` (Full):** Favors **rapid specialization** and captures nuances faster.

### Subsumption Toggle
Use `--no_subsumption` when you want to disable classifier subsumption entirely. This is useful for:
*   performance diagnostics
*   ablation studies
*   comparing canonical ACS2 behavior against a faster no-subsumption variant

The flag applies to both CPU and GPU backends, including hybrid runs and the maze benchmark helper.

## Dashboards and Metrics

The framework automatically generates a `_all.png` dashboard in the `reports/` folder after every run.

### Key Metrics:
*   **Knowledge (%):** Coverage of the state-action space by reliable rules.
*   **Generalization (%):** Usage of wildcards (`#`) to represent logic.
*   **Steps to Goal:** Efficiency of the learned policy.
*   **Population (Micro/Macro):** Total and unique rule counts.

## Documentation

For a full list of all available command-line parameters and detailed implementation notes, see [parameter_guide.md](parameter_guide.md).

## Citation

If you use this framework or any part of the code in your research or projects, please provide a proper citation to this repository:

```text
[Your Name/Team], "ACS2 Unified CPU/GPU Framework", GitHub Repository, 2026. 
URL: https://github.com/your-username/acs2-project
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, copy, modify, merge, publish, and distribute the software for all purposes, provided that the above **Citation** is included in all significant copies or substantial portions of the software.
