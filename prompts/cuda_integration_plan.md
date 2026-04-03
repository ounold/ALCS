# CUDA Integration Plan for ACS2 Project

This document outlines the strategy for integrating CUDA acceleration into the ACS2 experiment runner to improve performance, especially for large populations and high numbers of experiments.

## Current State Analysis
- **Execution Model:** `mainCPU.py` uses `multiprocessing` to run multiple independent experiments in parallel across CPU cores.
- **ACS2 Logic:** The core algorithm (matching, ALP, RL, GA) is implemented using Python objects (`Classifier`) and list/dictionary operations.
- **Bottlenecks:**
    - **Matching:** Iterating through the population to find classifiers matching the current state.
    - **ALP/RL Updates:** Serial updates of classifier parameters (q, r, fitness).
    - **Population Management:** Subsumption and duplicate checks involve nested loops or repeated comparisons.

## CUDA Integration Strategy

### 1. Vectorized Population (Tensor-based ACS2)
Instead of a list of `Classifier` objects, the population should be represented as a set of synchronized tensors (using PyTorch or CuPy).

| Attribute | Tensor Type | Shape | Description |
|-----------|-------------|-------|-------------|
| `C` (Condition) | `int8` | `(N, L)` | Matrix of attribute values (using a special code for '#'). |
| `A` (Action) | `int64` | `(N,)` | Vector of actions. |
| `E` (Effect) | `int8` | `(N, L)` | Matrix of anticipated attribute changes. |
| `q` (Quality) | `float32` | `(N,)` | Vector of classifier qualities. |
| `r` (Reward) | `float32` | `(N,)` | Vector of reward predictions. |
| `num` (Numerosity) | `int32` | `(N,)` | Vector of classifier copies. |
| `exp` (Experience) | `int32` | `(N,)` | Vector of experience counts. |

### 2. GPU-Accelerated Core Operations

#### A. Parallel Matching
Matching can be performed as a single tensor operation:
- Convert the current environment state to a tensor.
- Use broadcasting and logical operations to find indices of classifiers where `(C == state) | (C == '#')`.
- This operation is performed in parallel for all classifiers on the GPU.

#### B. Batch Reinforcement Learning
Updates for `q`, `r`, and `fitness` can be expressed as element-wise tensor operations applied to the `Action Set` indices.

#### C. ALP via Masking
Anticipatory Learning Process (ALP) can be implemented using boolean masks:
- Identify "correct" vs "incorrect" anticipators using tensor comparisons.
- Apply quality updates to both groups simultaneously.
- Generate new classifiers by applying offsets to existing tensors.

### 3. Parallel Across Runs (GPU)
Replace CPU `multiprocessing` with a single GPU-resident execution loop that handles all `N` experiments simultaneously.
- **State Tensor:** Shape `(N_exp, L)` representing the current state of all concurrent experiments.
- **Population Tensor:** A 3D tensor `(N_exp, Max_Pop, L)` or a flat representation with indexing to handle variable population sizes across experiments.
- **Execution:** Each step of the experiment is performed for all `N_exp` environments in one "pass" through the GPU kernels.

### 4. Hybrid Implementation Plan

#### Phase 1: Prototype Vectorized Matching
- Implement a `CudaACS2` class that maintains the population as PyTorch tensors.
- Implement the `match` function using PyTorch operations.
- Keep the environment interaction on the CPU (for now).

#### Phase 2: Vectorized Learning
- Port `_apply_rl` and `_apply_alp` to use tensor operations.
- Implement a basic GA using GPU-based sorting and selection.

#### Phase 3: Massively Parallel Experiments
- Refactor `mainCPU.py` into `mainGPU.py`.
- Move the Environment (GridWorld) logic to the GPU to avoid CPU-GPU data transfer overhead.
- Run hundreds of experiments in parallel by batching them like images in a CNN.

## Potential Challenges
- **Variable Population Size:** Tensors have fixed shapes. This can be handled by using a large pre-allocated buffer with "active" flags or using ragged tensors (if supported).
- **Control Flow:** ACS2 has complex logic (e.g., GA, subsumption) that can lead to "branch divergence" on the GPU. Minimizing conditional logic in favor of masked operations is key.
- **Memory Management:** For very large numbers of experiments, GPU VRAM might become a constraint.

## Recommended Tools
- **PyTorch:** Excellent for tensor operations and has built-in CUDA support.
- **CuPy:** A NumPy-compatible library for NVIDIA GPUs, useful if existing logic is heavily NumPy-based.
- **Numba:** Can be used to write custom CUDA kernels for specific ACS2 operations that are hard to vectorize.
