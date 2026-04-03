# CPU3 and GPU4 Harmonization Plan

## Goal

Bring the current CPU3 and GPU4 execution paths to the point where they are intentionally equivalent in:

- ACS2 learning semantics
- action selection behavior
- ALP and GA behavior
- population management
- metric collection
- summary reporting
- hybrid handoff semantics

The target is not "identical implementation style". The target is "same algorithmic behavior and same externally visible results within expected stochastic tolerance".

## Scope Compared

Compared files:

- `src/models/acs2/acs2CPU3.py`
- `src/models/acs2/acs2GPU4.py`
- `src/models/acs2/logicCPU3.py`
- `src/experiment_runnerCPU3.py`
- `src/experiment_runnerGPU4.py`
- `src/metricsCPU3.py`
- `src/metricsGPU4.py`
- `src/universal_runner.py`
- `src/models/acs2/hybrid_transfer.py`
- `environment/runtime_cpu3.py`
- `environment/runtime_gpu4.py`

## Key Discrepancies

### 1. CPU logical time does not advance, GPU time does

Observed:

- CPU initializes `self.time = 0` and uses it in ALP/GA bookkeeping, but never increments it in the CPU3 path.
  - `src/models/acs2/acs2CPU3.py:17`
  - `src/models/acs2/acs2CPU3.py:102`
  - `src/models/acs2/acs2CPU3.py:105`
- GPU increments `self.time` for active experiments on each learning step.
  - `src/models/acs2/acs2GPU4.py:116`

Impact:

- GA trigger timing diverges.
- `t_ga`, `t_alp`, and `aav` are effectively broken on CPU.
- classifier creation timestamps and age-dependent logic are not comparable.

Priority: P0

### 2. Exploration policy is materially different

Observed:

- CPU exploration is per-step and uses a two-branch policy:
  - with probability `epsilon`, choose a known action half the time and a fully random action half the time.
  - `src/models/acs2/acs2CPU3.py:51`
- GPU exploration uses a single scalar random draw for the whole batch and, when exploring, picks a fully random action for every experiment.
  - `src/models/acs2/acs2GPU4.py:100`

Impact:

- Different effective behavior even with same seed/config.
- GPU batch members are spuriously correlated.
- Coverage and learning speed differ.

Priority: P0

### 3. Action-value aggregation differs

Observed:

- CPU action choice uses max classifier fitness per action.
  - `src/models/acs2/acs2CPU3.py:46`
- GPU also uses max per action in `run_step`.
  - `src/models/acs2/acs2GPU4.py:98`
- But GPU `action_fitness_for_states()` sums fitness across matching rules instead of taking max.
  - `src/models/acs2/acs2GPU4.py:79`

Impact:

- Policy-map or evaluation helpers can disagree with runtime action choice.
- Hidden divergence risk for downstream utilities.

Priority: P1

### 4. Action-set subsumption is enabled on CPU but disabled on GPU

Observed:

- CPU adds with subsumption when `do_subsumption` is enabled.
  - `src/models/acs2/acs2CPU3.py:168`
  - `src/models/acs2/logicCPU3.py:140`
- GPU explicitly comments out action-set subsumption in learning.
  - `src/models/acs2/acs2GPU4.py:131`
- GPU still performs candidate-time subsumption during add, but not the same runtime action-set subsumption behavior.
  - `src/models/acs2/acs2GPU4.py:202`
  - `src/models/acs2/acs2GPU4.py:244`

Impact:

- Population size, numerosity growth, and generalization drift between backends.
- CPU and GPU are not running the same ACS2 variant.

Priority: P0

### 5. Population overflow policy differs sharply

Observed:

- CPU enforces `theta_as` by repeatedly deleting the lowest-`q` victim from the current action set.
  - `src/models/acs2/acs2CPU3.py:77`
  - `src/models/acs2/logicCPU3.py:99`
- GPU enforces `theta_as` later via batched `_control_action_set_size`.
  - `src/models/acs2/acs2GPU4.py:139`
  - `src/models/acs2/acs2GPU4.py:414`
- GPU also silently evicts globally low-`q` active classifiers when free tensor slots are exhausted during `batch_add_classifiers`.
  - `src/models/acs2/acs2GPU4.py:224`

Impact:

- GPU can delete classifiers unrelated to the current action set.
- `max_population` overflow produces semantics CPU does not have.
- Backends diverge most under population pressure.

Priority: P0

### 6. GA parent/offspring semantics are not equivalent

Observed:

- CPU selects parents from the action set using `q ** 3`, copies both parents, preserves each parent's action/effect, optionally crossovers conditions, mutates, and inserts two children.
  - `src/models/acs2/logicCPU3.py:29`
  - `src/models/acs2/logicCPU3.py:50`
- GPU samples two parent indices using a batched multinomial over `q ** 3`.
  - `src/models/acs2/acs2GPU4.py:393`
- GPU assigns both children `action_one`, so the second child's action is not inherited from parent two.
  - `src/models/acs2/acs2GPU4.py:398`
  - `src/models/acs2/acs2GPU4.py:410`
- GPU does not update parent `t_ga` after GA firing the way earlier GPU versions did, and unlike the classic CPU flow where new children get `t_ga=agent.time`.

Impact:

- Offspring distribution differs.
- Cross-action inheritance behavior differs.
- GA cadence may differ even after CPU time is fixed.

Priority: P0

### 7. ALP expected/unexpected paths are only approximately matched

Observed:

- CPU ALP operates classifier-by-classifier with immediate child creation/removal in loop order.
  - `src/models/acs2/acs2CPU3.py:95`
- GPU ALP computes batched masks and candidate lists, then applies them later.
  - `src/models/acs2/acs2GPU4.py:318`
- CPU removes low-quality classifiers during iteration and also mutates `action_set`.
  - `src/models/acs2/acs2CPU3.py:116`
- GPU applies a single global mask-based removal after candidate generation.
  - `src/models/acs2/acs2GPU4.py:376`

Impact:

- Same high-level intent, but ordering-sensitive edge cases can diverge.
- Candidate deduplication and deletion interactions may not match CPU.

Priority: P1

### 8. Metrics collection cadence differs

Observed:

- CPU only writes metric values on calculation episodes; non-sampled episodes remain zero.
  - `src/experiment_runnerCPU3.py:110`
- GPU caches the most recent metric values and fills every non-sampled episode with those cached values.
  - `src/experiment_runnerGPU4.py:141`

Impact:

- Episode series shape is the same, but meaning differs.
- Dashboard plots and summary means are not directly comparable.

Priority: P0

### 9. Creation-distribution metrics differ substantially

Observed:

- CPU captures creation distributions at 10 observation points across the run.
  - `src/experiment_runnerCPU3.py:132`
- GPU computes creation distribution as a single row at observation index `0` only.
  - `src/metricsGPU4.py:107`
  - `src/metricsGPU4.py:132`
- GPU writes the full creation tensor every time metrics are calculated, rather than sampling on observation points.
  - `src/experiment_runnerGPU4.py:131`

Impact:

- Creation-distribution plots are not semantically comparable.
- Hybrid merged dashboards privilege whichever backend ran last.

Priority: P0

### 10. Best-agent selection semantics differ

Observed:

- CPU returns the last experiment's agent as `best_agent`, not necessarily the best.
  - `src/experiment_runnerCPU3.py:209`
- GPU always returns experiment `0`.
  - `src/models/acs2/acs2GPU4.py:431`
- Universal runner then replicates the CPU best agent across all experiments for subsequent CPU phases.
  - `src/universal_runner.py:127`

Impact:

- Hybrid handoff is lossy and inconsistent.
- Dashboard "best agent" is arbitrary in both modes.

Priority: P1

### 11. CPU-to-GPU handoff is unsupported, GPU-to-CPU handoff is lossy

Observed:

- Universal runner explicitly states GPU runner does not support `initial_agents`.
  - `src/universal_runner.py:93`
- After a CPU phase, the runner stores `[best] * n_exp` rather than each experiment's evolved agent.
  - `src/universal_runner.py:127`
- GPU-to-CPU transfer merges duplicate imported classifiers by adding numerosity.
  - `src/models/acs2/hybrid_transfer.py:62`

Impact:

- Hybrid mode is not symmetric.
- Per-experiment diversity is discarded.
- Re-entry into GPU phases cannot preserve learned state.

Priority: P1

### 12. Summary statistics are not aligned

Observed:

- CPU summary omits reward metrics in `run_experimentCPU3`.
  - `src/experiment_runnerCPU3.py:219`
- GPU summary also omits reward metrics, but uses a different averaging convention for time and fills metrics differently.
  - `src/experiment_runnerGPU4.py:176`
- Hybrid summary later computes reward metrics from merged stats, but base runner summaries are inconsistent.
  - `src/hybrid_utils.py:154`

Impact:

- Single-backend reports are less comparable than hybrid reports.
- Time reporting is especially mismatched: CPU averages per-process durations, GPU uses wall time divided by `n_exp`.

Priority: P2

### 13. Environment execution semantics are mostly aligned, but batching changes behavior in exhaustive sampling

Observed:

- CPU exhaustive binary environments advance one sample per episode.
  - `environment/runtime_cpu3.py:143`
- GPU exhaustive binary environments advance `n_exp` samples per reset.
  - `environment/runtime_gpu4.py:150`

Impact:

- This is expected for batching, but it means experiment streams are not directly episode-aligned between CPU and GPU.
- Needs to be documented and excluded from strict equivalence tests unless explicitly normalized.

Priority: P3

## Harmonization Strategy

### Phase 1: Define the canonical semantics

Decide and document one canonical ACS2 behavior for both backends:

1. Exploration semantics:
   - per-experiment Bernoulli draw
   - when exploring, define exactly whether policy is:
     - random-known-vs-random-any with 50/50 split, or
     - random-any only
2. Subsumption semantics:
   - candidate-time subsumption only, or
   - candidate-time plus action-set subsumption
3. Population overflow semantics:
   - action-set-local deletion only, or
   - global eviction allowed when `max_population` is hit
4. Metric cadence semantics:
   - sparse only on sample episodes, or
   - forward-fill cached values
5. Best-agent semantics:
   - best by exploit performance
   - best by final fitness summary
   - or explicit "representative agent" separate from "best agent"

Deliverable:

- new design note in `.prompts` or `docs/` called `acs2_backend_equivalence_spec.md`

### Phase 2: Fix correctness-critical mismatches first

1. Fix CPU time advancement.
   - Increment `agent.time` once per environment step in CPU runner or agent learning path.
   - Add a regression test proving `time`, `t_ga`, and `t_alp` advance.
2. Fix GPU exploration to be per-experiment and match canonical policy.
3. Fix GPU GA child action/effect inheritance.
   - Child two should not blindly inherit `action_one`.
4. Decide and implement subsumption parity.
   - Either enable action-set subsumption on GPU or disable it on CPU.
5. Make CPU and GPU metric cadence match exactly.

Success criteria:

- CPU and GPU no longer disagree on the core ACS2 loop semantics.

### Phase 3: Unify population management

1. Extract a backend-agnostic "population semantics" spec:
   - duplicate merge rules
   - numerosity increments
   - candidate insertion order
   - deletion order
2. Rework GPU `batch_add_classifiers()` overflow behavior.
   - Prefer action-set-local deletion if canonical semantics requires it.
   - If `max_population` must remain a hard tensor cap, surface it as a documented deviation and expose diagnostics.
3. Add invariant checks:
   - no negative `num`
   - no inactive classifier contributes metrics
   - action-set size control always leaves total numerosity `<= theta_as`

### Phase 4: Unify metrics and reporting

1. Make creation-distribution sampling identical.
   - GPU should capture the same 10 observation points as CPU.
   - or CPU should switch to a new shared helper that both backends use.
2. Move metric collection policy into shared utilities.
3. Make summary-stat computation shared.
4. Define a single best-agent selection function used by both runners.

Success criteria:

- dashboard series and summary fields mean the same thing in both modes

### Phase 5: Fix hybrid handoff semantics

1. Preserve per-experiment agents across CPU phases instead of `[best] * n_exp`.
2. Add CPU-to-GPU import path if hybrid mode will support switching back.
3. Preserve creation metadata, numerosity, marks, and timestamps exactly across handoffs.
4. Add round-trip tests:
   - GPU -> CPU
   - CPU -> GPU
   - GPU -> CPU -> GPU

### Phase 6: Build equivalence tests

Add tests that compare CPU and GPU on small deterministic problems.

Suggested test matrix:

1. Grid maze, tiny map, `n_exp=2`, `n_steps=5`
2. Multiplexer, `n_steps=1`
3. ALP on, GA off
4. ALP on, GA on
5. subsumption on
6. low `max_population` stress case

For each run, compare:

- step counts
- action traces
- population macro/micro counts
- reliable counts
- knowledge
- generalization
- origin distributions
- creation distributions

Use tolerances only where stochasticity or floating-point batching truly requires them.

## Recommended Implementation Order

### Phase A: Canonical ACS2 semantics

1. CPU time advancement fix
2. GPU exploration parity
3. subsumption decision and parity
4. GPU population-control parity
5. GPU GA child inheritance and GA semantic parity
6. ALP semantic equivalence checks and fixes

### Phase B: Secondary equivalence issues

7. action-value aggregation parity for helper/evaluation paths
8. normalize environment-sampling assumptions in equivalence tests

### Phase C: Reporting and instrumentation

9. metric cadence parity
10. creation-distribution parity
11. shared summary utilities
12. best-agent selection refactor

### Phase D: Hybrid framework behavior

13. hybrid handoff preservation
14. CPU-to-GPU restore support

This order is intentionally derived from `.prompts/acs2_canonical_issues_from_pseudocode.md`, so canonical ACS2 mismatches are resolved before reporting and framework concerns.

## Concrete Refactor Targets

### Shared utilities to introduce

- `src/acs2_semantics.py`
  - exploration helper
  - best-agent selection helper
  - summary-stat helper
- `src/metrics_shared.py`
  - observation-point schedule
  - creation-distribution bucketing
- `src/handoff_shared.py`
  - explicit backend state serialization schema

### Existing files likely to change

- `src/models/acs2/acs2CPU3.py`
- `src/models/acs2/acs2GPU4.py`
- `src/models/acs2/logicCPU3.py`
- `src/experiment_runnerCPU3.py`
- `src/experiment_runnerGPU4.py`
- `src/metricsCPU3.py`
- `src/metricsGPU4.py`
- `src/universal_runner.py`
- `src/models/acs2/hybrid_transfer.py`

## Validation Checklist

- CPU and GPU both advance logical time consistently.
- CPU and GPU use the same exploration semantics.
- CPU and GPU use the same subsumption semantics.
- CPU and GPU create equivalent offspring in GA.
- CPU and GPU sample metrics on the same episodes.
- CPU and GPU creation distributions have the same observation-axis meaning.
- best-agent selection is deterministic and shared.
- hybrid transfer preserves per-experiment state.
- dashboard plots from CPU-only and GPU-only runs are comparable without special interpretation.

## Notes on Intentional Non-Equivalence

Some differences may remain intentional if documented:

- GPU may require `max_population` as a hard tensor cap.
- GPU batched exhaustive sampling may enumerate multiple inputs per episode reset.
- wall-clock timing metrics will never be directly comparable between multiprocessing CPU and batched GPU.

If any of these remain, they should be documented as explicit backend constraints, not accidental behavioral drift.
