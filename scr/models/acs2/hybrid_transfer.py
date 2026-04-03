from __future__ import annotations

from typing import List

from .acs2CPU3 import ACS2CPU3
from .acs2GPU4 import ACS2GPU4
from .classifierCPU3 import ClassifierCPU3
from .confCPU3 import ACS2ConfigurationCPU3
from .confGPU4 import ACS2ConfigurationGPU4


def _decode_symbols(values) -> List[str]:
    return ["#" if int(value) < 0 else str(int(value)) for value in values.tolist()]


def _copy_marks(mark_tensor) -> List[set[str]]:
    mark_sets: List[set[str]] = []
    for feature_marks in mark_tensor.tolist():
        current = {str(idx) for idx, enabled in enumerate(feature_marks) if enabled}
        mark_sets.append(current)
    return mark_sets


def gpu4_to_cpu3_agents(gpu_agent: ACS2GPU4, cpu_cfg: ACS2ConfigurationCPU3) -> List[ACS2CPU3]:
    if gpu_agent.l_len != cpu_cfg.l_len:
        raise ValueError("GPU4 and CPU3 configs must agree on state length")
    if gpu_agent.cfg.num_actions != cpu_cfg.num_actions:
        raise ValueError("GPU4 and CPU3 configs must agree on num_actions")
    if gpu_agent.symbol_capacity > cpu_cfg.symbol_capacity:
        raise ValueError("CPU3 symbol capacity must be at least GPU4 symbol capacity")

    imported_agents: List[ACS2CPU3] = []
    for experiment_index in range(gpu_agent.n_exp):
        cpu_agent = ACS2CPU3(cpu_cfg)
        cpu_agent.time = int(gpu_agent.time[experiment_index].item())
        cpu_agent.curr_ep_idx = int(gpu_agent.curr_ep_idx)
        active_indices = gpu_agent.active_mask[experiment_index].nonzero(as_tuple=False).flatten().tolist()
        for classifier_index in active_indices:
            imported = ClassifierCPU3(
                condition=_decode_symbols(gpu_agent.C[experiment_index, classifier_index]),
                action=int(gpu_agent.A[experiment_index, classifier_index].item()),
                effect=_decode_symbols(gpu_agent.E[experiment_index, classifier_index]),
                cfg=cpu_cfg,
                time_created=int(gpu_agent.t_ga[experiment_index, classifier_index].item()),
                origin_source=gpu_agent.reverse_origin_map.get(
                    int(gpu_agent.origin_source[experiment_index, classifier_index].item()),
                    "unknown",
                ),
                creation_episode=int(gpu_agent.creation_episode[experiment_index, classifier_index].item()),
            )
            imported.q = float(gpu_agent.q[experiment_index, classifier_index].item())
            imported.r = float(gpu_agent.r[experiment_index, classifier_index].item())
            imported.ir = float(gpu_agent.ir[experiment_index, classifier_index].item())
            imported.exp = int(gpu_agent.exp[experiment_index, classifier_index].item())
            imported.num = int(gpu_agent.num[experiment_index, classifier_index].item())
            imported.t_ga = int(gpu_agent.t_ga[experiment_index, classifier_index].item())
            imported.t_alp = int(gpu_agent.t_alp[experiment_index, classifier_index].item())
            imported.aav = float(gpu_agent.aav[experiment_index, classifier_index].item())
            imported.M = _copy_marks(gpu_agent.M[experiment_index, classifier_index])

            existing = cpu_agent.population_dict.get(imported.key)
            if existing is not None:
                existing.num += imported.num
                continue
            cpu_agent.population_dict[imported.key] = imported
            cpu_agent.population_by_action[imported.A].append(imported)
        imported_agents.append(cpu_agent)
    return imported_agents


def cpu3_to_gpu4_agent(cpu_agents: List[ACS2CPU3], gpu_cfg: ACS2ConfigurationGPU4, device: str = "cpu") -> ACS2GPU4:
    if not cpu_agents:
        raise ValueError("cpu_agents must not be empty")

    gpu_agent = ACS2GPU4(gpu_cfg, len(cpu_agents), device=device)

    for experiment_index, cpu_agent in enumerate(cpu_agents):
        gpu_agent.time[experiment_index] = int(cpu_agent.time)
        gpu_agent.curr_ep_idx = max(gpu_agent.curr_ep_idx, int(cpu_agent.curr_ep_idx))
        population = list(cpu_agent.population_dict.values())
        if len(population) > gpu_agent.max_pop:
            raise ValueError(
                f"CPU agent {experiment_index} has {len(population)} unique classifiers, "
                f"which exceeds GPU max_population={gpu_agent.max_pop}"
            )

        for slot_index, classifier in enumerate(population):
            gpu_agent.C[experiment_index, slot_index] = _encode_symbols(classifier.C)
            gpu_agent.A[experiment_index, slot_index] = int(classifier.A)
            gpu_agent.E[experiment_index, slot_index] = _encode_symbols(classifier.E)
            gpu_agent.q[experiment_index, slot_index] = float(classifier.q)
            gpu_agent.r[experiment_index, slot_index] = float(classifier.r)
            gpu_agent.ir[experiment_index, slot_index] = float(classifier.ir)
            gpu_agent.exp[experiment_index, slot_index] = int(classifier.exp)
            gpu_agent.num[experiment_index, slot_index] = int(classifier.num)
            gpu_agent.aav[experiment_index, slot_index] = float(classifier.aav)
            gpu_agent.t_ga[experiment_index, slot_index] = int(classifier.t_ga)
            gpu_agent.t_alp[experiment_index, slot_index] = int(classifier.t_alp)
            gpu_agent.active_mask[experiment_index, slot_index] = True
            gpu_agent.origin_source[experiment_index, slot_index] = gpu_agent.origin_map.get(classifier.origin_source, 0)
            gpu_agent.creation_episode[experiment_index, slot_index] = int(classifier.creation_episode)
            gpu_agent.M[experiment_index, slot_index] = _encode_marks(classifier.M, gpu_cfg.symbol_capacity)

    return gpu_agent


def _encode_symbols(values: List[str]):
    import torch

    encoded = [-1 if value == "#" else int(value) for value in values]
    return torch.tensor(encoded, dtype=torch.long)


def _encode_marks(mark_sets: List[set[str]], symbol_capacity: int):
    import torch

    mark_tensor = torch.zeros((len(mark_sets), symbol_capacity), dtype=torch.bool)
    for feature_index, mark_set in enumerate(mark_sets):
        for value in mark_set:
            value_index = int(value)
            if 0 <= value_index < symbol_capacity:
                mark_tensor[feature_index, value_index] = True
    return mark_tensor
