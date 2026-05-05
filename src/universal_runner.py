from __future__ import annotations

import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.configGPU4 import ExperimentConfigGPU4
from src.experiment_runnerGPU4 import run_experimentGPU4
from src.experiment_runnerCPU3 import run_experimentCPU3
from src.hybrid_utils import gpu_config_to_cpu3, merge_experiment_stats, calculate_hybrid_summary
from src.models.acs2.hybrid_transfer import gpu4_to_cpu3_agents, cpu3_to_gpu4_agent
from src.models.acs2.acs2GPU4 import ACS2GPU4, AgentSelectionGPU4
from src.models.acs2.confGPU4 import ACS2ConfigurationGPU4

class UniversalRunner:
    def __init__(
        self,
        config: ExperimentConfigGPU4,
        explore_mode: str,
        exploit_mode: str,
        gpu_agent_cfg_override: Optional[ACS2ConfigurationGPU4] = None,
    ):
        self.config = config
        self.explore_mode = explore_mode
        self.exploit_mode = exploit_mode
        self.gpu_agent_cfg_override = gpu_agent_cfg_override
        self.current_agents = None
        self.stats_history = []
        self.last_env = None
        self.best_agent = None
        self.episode_offset = 0

    @staticmethod
    def _is_gpu_mode(mode: str) -> bool:
        return mode in {"gpu", "gpu_seq"}

    def run(self) -> Tuple[Dict[str, Any], Dict[str, Any], Any, Any, float]:
        """
        Runs all phases (explore, exploit1, exploit2) using the specified modes.
        """
        import time
        from src.experiment_runnerCPU3 import create_environmentCPU3, calculate_optimal_metricsCPU3
        
        # Initial capture of optimal metrics to avoid 0.0 if phases are skipped
        dummy_env = create_environmentCPU3(self.config.environment)
        self.optimal_avg_steps = calculate_optimal_metricsCPU3(dummy_env)

        phases = ["explore", "exploit1", "exploit2"]
        total_wall_time = 0
        phase_summaries = []
        actual_explore_episodes_run = 0
        timing_breakdown: Dict[str, float] = {}
        
        for phase_name in phases:
            mode = self.explore_mode if phase_name == "explore" else self.exploit_mode
            phase_config = self.config.phases[phase_name]
            
            if phase_config.episodes <= 0:
                continue

            print(f"--- Starting Phase: {phase_name} (Mode: {mode}, Episodes: {phase_config.episodes}) ---")
            
            start_t = time.time()
            if self._is_gpu_mode(mode):
                stats, summary, completed, best, opt, env = self._run_gpu_phase(phase_name, mode)
            else:
                no_mp = (mode == "cpu_single")
                stats, summary, completed, best, opt, env = self._run_cpu_phase(phase_name, no_mp)
            
            duration = time.time() - start_t
            total_wall_time += duration
            phase_summaries.append(summary)
            phase_timing = summary.get("Timing Breakdown", {})
            for key, value in phase_timing.items():
                timing_breakdown[key] = timing_breakdown.get(key, 0.0) + float(value)

            self.stats_history.append(stats)
            self.episode_offset += completed 
            self.last_env = env
            self.best_agent = best 
            self.optimal_avg_steps = opt # Captured from runner
            
            if phase_name == "explore":
                actual_explore_episodes_run = completed # Record actual explore episodes run

        merged_stats = merge_experiment_stats(self.stats_history)
        final_summary = calculate_hybrid_summary(
            merged_stats,
            actual_explore_episodes_run,
            self.config.phases["exploit1"].episodes,
        )
        
        # Aggregate Time Metrics
        final_summary["Total Time"] = total_wall_time
        if phase_summaries:
            final_summary["Avg Time"] = float(sum(float(summary.get("Avg Time", 0.0)) for summary in phase_summaries))
            phase_variance = sum(float(summary.get("Std Time", 0.0)) ** 2 for summary in phase_summaries)
            final_summary["Std Time"] = float(np.sqrt(phase_variance))
            final_summary["ALP Marking"] = phase_summaries[-1].get("ALP Marking", phase_summaries[0].get("ALP Marking"))
            final_summary["Subsumption"] = phase_summaries[-1].get("Subsumption", phase_summaries[0].get("Subsumption"))
        else:
            final_summary["Avg Time"] = 0.0
            final_summary["Std Time"] = 0.0
        if timing_breakdown:
            final_summary["Timing Breakdown"] = timing_breakdown

        merged_stats["best_agent"] = self.best_agent
        
        return merged_stats, final_summary, self.best_agent, self.last_env, self.optimal_avg_steps

    def _run_gpu_phase(self, phase_name: str, mode: str):
        if mode == "gpu_seq":
            return self._run_gpu_seq_phase(phase_name)

        # GPU runner runs exactly one phase if stop_after_phase is set
        # However, it expects a full config. We must ensure other phases are 0'd out
        # or it will run them all.
        gpu_cfg = self._get_single_phase_gpu_config(phase_name)
        from src.hybrid_utils import build_agent_configGPU4
        agent_cfg_gpu = self.gpu_agent_cfg_override or build_agent_configGPU4(gpu_cfg)
        initial_gpu_agent = None
        if self.current_agents is not None:
            if isinstance(self.current_agents, list):
                if self.current_agents and hasattr(self.current_agents[0], "n_exp"):
                    initial_gpu_agent = self._combine_single_gpu_agents(self.current_agents, agent_cfg_gpu, gpu_cfg.device)
                else:
                    initial_gpu_agent = cpu3_to_gpu4_agent(self.current_agents, agent_cfg_gpu, device=gpu_cfg.device)
            elif hasattr(self.current_agents, "n_exp"):
                initial_gpu_agent = self.current_agents
                initial_gpu_agent.cfg = agent_cfg_gpu
            else:
                initial_gpu_agent = cpu3_to_gpu4_agent(self.current_agents, agent_cfg_gpu, device=gpu_cfg.device)
        
        stats, summary, completed, best, opt, env = run_experimentGPU4(
            gpu_cfg, 
            agent_cfg_gpu,
            stop_after_phase=phase_name,
            initial_agent=initial_gpu_agent,
            episode_offset=self.episode_offset,
        )
        self.current_agents = stats.get("all_agents", best.agent if best is not None else None)
        return stats, summary, completed, best, opt, env

    def _run_cpu_phase(self, phase_name: str, no_mp: bool):
        cpu_cfg = gpu_config_to_cpu3(self.config, [phase_name])
        
        # Build initial agents if coming from GPU
        initial_cpu_agents = None
        if self.current_agents is not None:
            from mainCPU3 import build_agent_configCPU3
            agent_cfg_cpu = build_agent_configCPU3(cpu_cfg)
            if isinstance(self.current_agents, list):
                if self.current_agents and hasattr(self.current_agents[0], "n_exp"):
                    initial_cpu_agents = [gpu4_to_cpu3_agents(agent, agent_cfg_cpu)[0] for agent in self.current_agents]
                else:
                    initial_cpu_agents = self.current_agents
            else:
                # If current_agents is a GPU AgentSelectionGPU4, convert it
                if hasattr(self.current_agents, 'n_exp'):
                    initial_cpu_agents = gpu4_to_cpu3_agents(self.current_agents, agent_cfg_cpu)
                else:
                    initial_cpu_agents = self.current_agents

        stats, summary, completed, best, opt, env = run_experimentCPU3(
            cpu_cfg,
            None, # agent_cfg built inside
            no_mp=no_mp,
            initial_agents=initial_cpu_agents,
            episode_offset=self.episode_offset
        )
        self.current_agents = stats.get("all_agents", initial_cpu_agents)
        return stats, summary, completed, best, opt, env

    def _run_gpu_seq_phase(self, phase_name: str):
        gpu_cfg = self._get_single_phase_gpu_config(phase_name, n_exp_override=1)
        from src.hybrid_utils import build_agent_configGPU4

        agent_cfg_gpu = self.gpu_agent_cfg_override or build_agent_configGPU4(gpu_cfg)
        initial_gpu_agents: Optional[List[ACS2GPU4]] = None
        if self.current_agents is not None:
            if isinstance(self.current_agents, list):
                if self.current_agents and hasattr(self.current_agents[0], "n_exp"):
                    initial_gpu_agents = self.current_agents
                else:
                    initial_gpu_agents = [cpu3_to_gpu4_agent([cpu_agent], agent_cfg_gpu, device=gpu_cfg.device) for cpu_agent in self.current_agents]
            elif hasattr(self.current_agents, "n_exp"):
                initial_gpu_agents = self._split_batched_gpu_agent(self.current_agents, agent_cfg_gpu, gpu_cfg.device)

        stats_runs: List[Dict[str, Any]] = []
        durations: List[float] = []
        timing_breakdown: Dict[str, float] = {}
        resulting_agents: List[ACS2GPU4] = []
        best_agent = None
        best_score = float("inf")
        env = None
        opt = self.optimal_avg_steps
        completed = 0

        for exp_idx in range(self.config.n_exp):
            single_gpu_cfg = self._get_single_phase_gpu_config(
                phase_name,
                n_exp_override=1,
                seed_override=self.config.seed + exp_idx,
            )
            initial_agent = initial_gpu_agents[exp_idx] if initial_gpu_agents is not None and exp_idx < len(initial_gpu_agents) else None
            stats, summary, completed, best, opt, env = run_experimentGPU4(
                single_gpu_cfg,
                agent_cfg_gpu,
                stop_after_phase=phase_name,
                initial_agent=initial_agent,
                episode_offset=self.episode_offset,
            )
            stats_runs.append(stats)
            durations.append(float(summary.get("Total Time", 0.0)))
            phase_timing = summary.get("Timing Breakdown", {})
            for key, value in phase_timing.items():
                timing_breakdown[key] = timing_breakdown.get(key, 0.0) + float(value)
            if hasattr(stats.get("all_agents"), "n_exp"):
                resulting_agents.append(stats["all_agents"])
            elif best is not None:
                resulting_agents.append(best.agent)

            candidate_gpu = resulting_agents[-1] if resulting_agents else None
            if candidate_gpu is not None:
                steps_key = stats.get("stats_steps")
                if isinstance(steps_key, np.ndarray) and steps_key.size > 0:
                    score = float(np.mean(steps_key))
                else:
                    score = float("inf")
                if score < best_score:
                    best_score = score
                    best_agent = AgentSelectionGPU4(candidate_gpu, 0)

        merged_stats = self._merge_stats_across_experiments(stats_runs)
        merged_stats["all_agents"] = resulting_agents
        summary = {
            "Total Time": float(sum(durations)),
            "Avg Time": float(np.mean(durations)) if durations else 0.0,
            "Std Time": float(np.std(durations)) if len(durations) > 1 else 0.0,
            "ALP Marking": "Restricted (on incorrect)" if agent_cfg_gpu.alp_mark_only_incorrect else "Full Action Set",
            "Subsumption": "Disabled" if not agent_cfg_gpu.do_subsumption else "Enabled",
        }
        phase_summary = calculate_hybrid_summary(
            merged_stats,
            self._phase_episode_count("explore", phase_name),
            self._phase_episode_count("exploit1", phase_name),
        )
        summary.update(phase_summary)
        if timing_breakdown:
            summary["Timing Breakdown"] = timing_breakdown

        self.current_agents = resulting_agents
        return merged_stats, summary, completed, best_agent, opt, env

    def _phase_episode_count(self, queried_phase: str, active_phase: str) -> int:
        return self.config.phases[queried_phase].episodes if queried_phase == active_phase else 0

    def _merge_stats_across_experiments(self, stats_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not stats_runs:
            return {}
        merged: Dict[str, Any] = {}
        array_keys = [key for key, value in stats_runs[0].items() if isinstance(value, np.ndarray)]
        for key in array_keys:
            values = [run[key] for run in stats_runs if key in run]
            if not values:
                continue
            if values[0].ndim == 2:
                merged[key] = np.concatenate(values, axis=0)
            elif values[0].ndim == 3:
                merged[key] = np.concatenate(values, axis=0)
            else:
                merged[key] = np.concatenate(values, axis=0)

        if "stats_steps" in merged:
            merged["mean_steps"] = np.mean(merged["stats_steps"], axis=0)

        mean_aliases = [
            "micro", "rmicro", "macro", "rmacro", "know", "avg_r", "avg_rel_r",
            "avg_q_all", "avg_q_rel", "avg_fit_all", "avg_fit_rel", "generalization",
            "covering_perc", "ga_perc", "alp_unexpected_perc", "alp_expected_perc", "alp_covering_perc",
            "covering_abs", "ga_abs", "alp_unexpected_abs", "alp_expected_abs", "alp_covering_abs",
        ]
        for name in mean_aliases:
            stats_key = f"stats_{name}"
            if stats_key in merged:
                merged[f"mean_{name}"] = np.mean(merged[stats_key], axis=0)
        if "stats_know" in merged:
            merged["std_know"] = np.std(merged["stats_know"], axis=0)
        for origin in ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]:
            dist_key = f"stats_{origin}_creation_dist"
            abs_key = f"stats_{origin}_creation_dist_abs"
            if dist_key in merged:
                merged[f"mean_{origin}_creation_dist"] = np.mean(merged[dist_key], axis=0)
                merged[f"{origin}_creation_dist"] = merged[f"mean_{origin}_creation_dist"]
            if abs_key in merged:
                merged[f"mean_{origin}_creation_dist_abs"] = np.mean(merged[abs_key], axis=0)
                merged[f"{origin}_creation_dist_abs"] = merged[f"mean_{origin}_creation_dist_abs"]

        if "mean_micro" in merged:
            merged["mean_micro_pop"] = merged["mean_micro"]
        if "mean_rmicro" in merged:
            merged["mean_rel_micro_pop"] = merged["mean_rmicro"]
        if "mean_macro" in merged:
            merged["mean_macro_pop"] = merged["mean_macro"]
        if "mean_rmacro" in merged:
            merged["mean_rel_macro_pop"] = merged["mean_rmacro"]
        return merged

    def _split_batched_gpu_agent(self, batched_agent: ACS2GPU4, agent_cfg_gpu: ACS2ConfigurationGPU4, device: str) -> List[ACS2GPU4]:
        single_agents: List[ACS2GPU4] = []
        for exp_idx in range(batched_agent.n_exp):
            single_agent = ACS2GPU4(agent_cfg_gpu, 1, device=device)
            for attr_name in ("C", "A", "E", "q", "r", "ir", "exp", "num", "aav", "t_ga", "t_alp", "M", "active_mask", "origin_source", "creation_episode"):
                source = getattr(batched_agent, attr_name)
                target = getattr(single_agent, attr_name)
                target[0].copy_(source[exp_idx])
            single_agent.time[0] = batched_agent.time[exp_idx]
            single_agent.curr_ep_idx = batched_agent.curr_ep_idx
            single_agents.append(single_agent)
        return single_agents

    def _combine_single_gpu_agents(self, single_agents: List[ACS2GPU4], agent_cfg_gpu: ACS2ConfigurationGPU4, device: str) -> ACS2GPU4:
        batched_agent = ACS2GPU4(agent_cfg_gpu, len(single_agents), device=device)
        batched_agent.curr_ep_idx = max((agent.curr_ep_idx for agent in single_agents), default=0)
        for exp_idx, single_agent in enumerate(single_agents):
            for attr_name in ("C", "A", "E", "q", "r", "ir", "exp", "num", "aav", "t_ga", "t_alp", "M", "active_mask", "origin_source", "creation_episode"):
                source = getattr(single_agent, attr_name)
                target = getattr(batched_agent, attr_name)
                target[exp_idx].copy_(source[0])
            batched_agent.time[exp_idx] = single_agent.time[0]
        return batched_agent

    def _get_single_phase_gpu_config(self, phase_name: str, n_exp_override: Optional[int] = None, seed_override: Optional[int] = None) -> ExperimentConfigGPU4:
        # Create a copy where only the target phase has episodes
        new_phases = {}
        for name, p in self.config.phases.items():
            eps = p.episodes if name == phase_name else 0
            new_phases[name] = copy.deepcopy(p)
            # We must trick the validator/runner by actually modifying the episodes
            # ExperimentConfigGPU4 is frozen, so we use type() trick or just pass to constructor
            object.__setattr__(new_phases[name], 'episodes', eps)
            
        return ExperimentConfigGPU4(
            n_exp=self.config.n_exp if n_exp_override is None else n_exp_override,
            seed=self.config.seed if seed_override is None else seed_override,
            n_steps=self.config.n_steps,
            beta=self.config.beta,
            gamma=self.config.gamma,
            theta_i=self.config.theta_i,
            theta_r=self.config.theta_r,
            epsilon=self.config.epsilon,
            u_max=self.config.u_max,
            theta_ga=self.config.theta_ga,
            mu=self.config.mu,
            chi=self.config.chi,
            theta_as=self.config.theta_as,
            theta_exp=self.config.theta_exp,
            alp_mark_only_incorrect=self.config.alp_mark_only_incorrect,
            no_subsumption=self.config.no_subsumption,
            metric_calculation_frequency=self.config.metric_calculation_frequency,
            max_population=self.config.max_population,
            environment=self.config.environment,
            phases=new_phases,
            device=self.config.device,
            steps_to_goal_threshold=self.config.steps_to_goal_threshold
        )
