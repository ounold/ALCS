from __future__ import annotations

import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.configGPU4 import ExperimentConfigGPU4
from src.experiment_runnerGPU4 import run_experimentGPU4
from src.experiment_runnerCPU3 import run_experimentCPU3
from src.hybrid_utils import gpu_config_to_cpu3, merge_experiment_stats, calculate_hybrid_summary
from src.models.acs2.hybrid_transfer import gpu4_to_cpu3_agents, cpu3_to_gpu4_agent
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
            if mode == "gpu":
                stats, summary, completed, best, opt, env = self._run_gpu_phase(phase_name)
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

    def _run_gpu_phase(self, phase_name: str):
        # GPU runner runs exactly one phase if stop_after_phase is set
        # However, it expects a full config. We must ensure other phases are 0'd out
        # or it will run them all.
        gpu_cfg = self._get_single_phase_gpu_config(phase_name)
        from src.hybrid_utils import build_agent_configGPU4
        agent_cfg_gpu = self.gpu_agent_cfg_override or build_agent_configGPU4(gpu_cfg)
        initial_gpu_agent = None
        if self.current_agents is not None:
            if hasattr(self.current_agents, "n_exp"):
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
            # If current_agents is a GPU AgentSelectionGPU4, convert it
            if hasattr(self.current_agents, 'n_exp'):
                initial_cpu_agents = gpu4_to_cpu3_agents(self.current_agents, agent_cfg_cpu)
            else:
                # Coming from a previous CPU phase, it's already a list of agents
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

    def _get_single_phase_gpu_config(self, phase_name: str) -> ExperimentConfigGPU4:
        # Create a copy where only the target phase has episodes
        new_phases = {}
        for name, p in self.config.phases.items():
            eps = p.episodes if name == phase_name else 0
            new_phases[name] = copy.deepcopy(p)
            # We must trick the validator/runner by actually modifying the episodes
            # ExperimentConfigGPU4 is frozen, so we use type() trick or just pass to constructor
            object.__setattr__(new_phases[name], 'episodes', eps)
            
        return ExperimentConfigGPU4(
            n_exp=self.config.n_exp,
            seed=self.config.seed,
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
