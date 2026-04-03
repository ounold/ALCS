# Genetic Algorithm Optimization Plan for ACS2 (CPU2)

This document outlines the strategy to accelerate the Genetic Algorithm (GA) and population management in the ACS2 implementation. The goal is to move from iterative list-based operations to vectorized bitwise operations and efficient data structures.

## 1. High-Performance Population Management

Currently, `add_to_population` performs a linear search ($O(N)$) for every new classifier. As the population grows, this becomes the primary bottleneck.

### Strategy:
- **Hashing for Duplicate Detection**: Store the population in a Python `dict` where the key is a tuple: `(condition_bits, wildcard_mask, action)`. This reduces duplicate checking to $O(1)$.
- **Action-Grouped Subsumption**: Maintain a secondary index (a dictionary of lists) grouping classifiers by their action. Since subsumption only occurs between classifiers with the same action, this significantly reduces the search space.
- **Lazy List Synchronization**: Only convert the population dictionary back to a list when required for matching (if not using vectorized matching) or reporting.

## 2. Bitwise Genetic Operators

Refactor crossover and mutation to operate directly on the `condition_bits` and `wildcard_mask` instead of the string-based list `C`.

### Implementation:
- **Bitwise Crossover**: Generate a random bitmask corresponding to the crossover point and use bitwise logic: `child = (p1 & mask) | (p2 & ~mask)`.
- **Bitwise Mutation**: Use bitwise `OR` to generalize bits (changing a specific value to a wildcard) or random mask generation to toggle bits.
- **Speedup**: This eliminates list slicing, joining, and integer parsing during every GA cycle.

## 3. Numba-Accelerated Subsumption

The `does_subsume` logic is called thousands of times per second. Even with bitmasks, the Python interpreter adds overhead.

### Implementation:
- **JIT Compilation**: Use the `numba` library to JIT-compile the `does_subsume` function.
- **Function Signature**: 
  ```python
  @numba.njit
  def fast_subsume(g_bits, g_mask, s_bits, s_mask):
      # Bitwise logic for consistency and generality
      pass
  ```
- **Benefit**: Compiles the logic to machine code, achieving performance near C++.

## 4. Vectorized Selection with NumPy

The current selection process (`select_offspring`) iterates through the action set to calculate weights and select parents.

### Strategy:
- **NumPy Random Choice**: Represent the action set qualities as a NumPy array and use `np.random.choice` with the `p` parameter (probabilities) for batch selection of parents.
- **Bulk Repopulation**: If multiple offspring are needed, perform selection and reproduction in a single vectorized pass.

## 5. Summary of Technologies to Use

| Technology | Purpose |
| :--- | :--- |
| **Python `dict`** | $O(1)$ duplicate checking via hashing. |
| **Bitwise Ops** | Constant-time crossover, mutation, and matching. |
| **Numba** | Machine-code speed for core logical checks (subsumption). |
| **NumPy** | Vectorized parent selection and quality updates. |

## 6. Validation and Benchmarking

1.  **Metric Consistency**: Ensure that bitwise crossover/mutation results in the same logical outcomes as string-based ones.
2.  **Profiling**: Use `cProfile` to compare the `ga_evolve` execution time before and after optimizations.
3.  **Scalability Test**: Run experiments with 10,000+ exploration episodes to verify that $O(1)$ hashing prevents the performance degradation typical of linear population searches.
