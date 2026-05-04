# Experiment Framework Documentation

> Note: the current ACS2 workflow in this repository is centered on `acs2.py` and the maze benchmark helper `run_maze_benchmarks.py`. In particular, the benchmark CSV now records exploit-phase quality, timing, and population statistics, including `micro_pop_exploit2_avg`, `macro_pop_exploit2_avg`, `micro_pop_rel_exploit2_avg`, and `macro_pop_rel_exploit2_avg`.

This document outlines the steps to define, run, and analyze experiments using the provided framework.

## 1. Defining an Experiment

Experiments are defined in YAML files located in the `experiments/configs/` directory.

### Configuration File Structure

Each configuration file has the following main sections:

-   `experiment_name`: A string to identify the experiment. This name is used for the output directory.
-   `repetitions`: The number of times to repeat the experiment for each model to gather statistical data.
-   `total_episodes`: The number of episodes to run for each repetition.
-   `environment`: Defines the environment to be used for the experiment.
    -   `name`: The name of the environment class (e.g., `GridEnvironment`).
    -   `params`: The parameters to be passed to the environment's constructor.
-   `models`: A list of models to be run in the experiment. Each model has:
    -   `model_name`: A unique name for this model instance in the experiment.
    -   `path`: The full Python path to the model's class (e.g., `src.models.q_learning.q_learning.QLearning`).
    -   `config_path`: The full Python path to the model's configuration class.
    -   `params`: The hyperparameters for the model.

### Example Configuration

Here is an example of an experiment configuration file (`experiments/configs/acs2_vs_qlearning.yaml`):

```yaml
experiment_name: ACS2_vs_QLearning
repetitions: 10
total_episodes: 500
environment:
  name: GridEnvironment
  params:
    rows: 7
    cols: 7
    start_pos: [0, 0]
    goal_pos: [0, 6]
    obstacles:
      - [1, 2]
      - [1, 4]
      # ...

models:
  - model_name: ACS2
    path: src.models.acs2.acs2.ACS2
    config_path: src.models.acs2.conf.ACS2Configuration
    params:
      beta: 0.1
      gamma: 0.95
      epsilon: 0.2

  - model_name: Q-learning
    path: src.models.q_learning.q_learning.QLearning
    config_path: src.models.q_learning.conf.QLearningConfiguration
    params:
      learning_rate: 0.1
      discount_factor: 0.9
      epsilon: 0.2
```

## 2. Running an Experiment

To run an experiment, use the `run_experiments.py` script with the path to your experiment configuration file.

### Command

```bash
python run_experiments.py experiments/configs/your_experiment_config.yaml
```
*(Note: You may need to use `python3` or specify the path to your Python executable, e.g., `D:\Documents\GitHub\notatnik\.venv1\Scripts\python`)*

### Output

The script will create a new directory for the results in `experiments/results/`. The directory will be named `<experiment_name>_<timestamp>`. Inside this directory, you will find a CSV file for each run of each model, containing the collected metrics.

## 3. Analyzing the Results

To compare the results of one or more experiments, use the `analysis/compare_experiments.py` script.

### Command

You can provide one or more result directories to the script.

**To analyze a single experiment:**
```bash
python analysis/compare_experiments.py experiments/results/<experiment_name>_<timestamp>
```

**To compare multiple experiments:**
```bash
python analysis/compare_experiments.py experiments/results/<exp1_dir> experiments/results/<exp2_dir>
```

*(Note: You may need to use `python3` or specify the path to your Python executable.)*

### Optional Arguments

-   `--metric`: The metric to plot and compare. The default is `steps_to_goal`.

### Output

The analysis script generates:
-   A plot (`comparison_<metric>.png`) showing the average performance of each model over time, with the standard deviation as a shaded area. The plot is saved in the experiment directory if only one is provided, or in the root directory if multiple are compared.
-   A statistical t-test comparing the final performance of the models (if two models are being compared).
