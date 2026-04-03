from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from environment.runtime_cpu3 import EnvironmentCPU3, GridEnvironmentCPU3, environment_from_metadataCPU3
from src.models.acs2.acs2CPU3 import ACS2CPU3


ARROW_LABELS_CPU3 = {0: "←", 1: "→", 2: "↑", 3: "↓", 4: "↖", 5: "↗", 6: "↙", 7: "↘"}

ARROW_LABELS_CPU3 = {0: "\u2190", 1: "\u2192", 2: "\u2191", 3: "\u2193", 4: "\u2196", 5: "\u2197", 6: "\u2199", 7: "\u2198"}


def _sanitize_title_prefixCPU3(title_prefix: str) -> str:
    cleaned = title_prefix.replace("Unified", "").replace("[no-subsumption]", "").replace("[subsumption]", "")
    return (" ".join(cleaned.split()) + " ") if cleaned.strip() else ""


def _alp_marking_labelCPU3(value: Any) -> str:
    if value in (True, "incorrect_only", "Restricted (on incorrect)"):
        return "Restricted (on incorrect)"
    if value in (False, "all_action_set", "Full Action Set"):
        return "Full Action Set"
    return str(value)


def calculate_emaCPU3(data: np.ndarray, span: int) -> np.ndarray:
    if len(data) == 0:
        return np.array([])
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for idx in range(1, len(data)):
        ema[idx] = alpha * data[idx] + (1 - alpha) * ema[idx - 1]
    return ema


def calculate_policy_avg_lenCPU3(agent: Optional[ACS2CPU3], env: EnvironmentCPU3) -> float:
    if not agent or not isinstance(env, GridEnvironmentCPU3):
        return 0.0
    total_steps = 0
    solved_cases = 0
    for row in range(env.rows):
        for col in range(env.cols):
            if (row, col) == env.goal_pos or (row, col) in env.obstacles:
                continue
            current = [row, col]
            visited = set()
            for steps in range(1, 51):
                current_tuple = tuple(current)
                if current_tuple in visited:
                    break
                visited.add(current_tuple)
                state_bits = agent.state_to_bits([str(current[0]), str(current[1])])
                match_set = [classifier for classifier in agent.population if classifier.matches_bits(state_bits)]
                if not match_set:
                    break
                action_fitness: Dict[int, float] = {}
                for classifier in match_set:
                    action_fitness[classifier.A] = max(classifier.fitness, action_fitness.get(classifier.A, -1.0))
                best_action = max(action_fitness, key=action_fitness.get)
                current = [int(value) for value in env.peek_step(current, best_action)]
                if tuple(current) == env.goal_pos:
                    total_steps += steps
                    solved_cases += 1
                    break
    return total_steps / solved_cases if solved_cases > 0 else 0.0


def calculate_exploit_avg_stdCPU3(stats: Dict[str, Any], params_phases: Dict[str, Any]) -> float:
    steps = np.asarray(stats.get("stats_steps", np.array([])), dtype=float)
    if steps.size == 0:
        return 0.0
    exploit2_start = int(params_phases.get("explore", {}).get("episodes", 0)) + int(params_phases.get("exploit1", {}).get("episodes", 0))
    exploit2_slice = steps[:, exploit2_start:] if steps.ndim == 2 else np.array([])
    if exploit2_slice.size == 0:
        exploit2_slice = steps
    return float(np.std(exploit2_slice))


def _plot_stepsCPU3(ax, mean_steps, ema_values, optimal_avg_steps, exploit_avg, boundary_one, boundary_two, ema_span, title_prefix=""):
    raw_line, = ax.plot(mean_steps, color="lightblue", label=f"Mean Steps (Exploit Avg.: {exploit_avg:.2f})")
    ema_line, = ax.plot(ema_values, color="red", linestyle="--", label=f"EMA {ema_span}")
    optimal_line = ax.axhline(y=optimal_avg_steps, color="magenta", linestyle="-.", label=f"Optimal: {optimal_avg_steps:.2f}")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    ax.set_title(f"{title_prefix}Steps to Goal")
    ax.legend(handles=[ema_line, raw_line, optimal_line], fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_populationCPU3(ax, mean_micro, mean_rel_micro, mean_macro, mean_rel_macro, boundary_one, boundary_two, title_prefix=""):
    line_micro, = ax.plot(mean_micro, color="green", label="Micro (All)")
    line_rel_micro, = ax.plot(mean_rel_micro, color="green", linestyle="--", label="Micro (Rel)", alpha=0.8)
    ax.set_ylabel("Micro-population Size", color="green")
    ax.tick_params(axis="y", labelcolor="green")
    twin = ax.twinx()
    line_macro, = twin.plot(mean_macro, color="blue", label="Macro (All)")
    line_rel_macro, = twin.plot(mean_rel_macro, color="blue", linestyle="--", label="Macro (Rel)")
    twin.set_ylabel("Macro-population Size", color="blue")
    twin.tick_params(axis="y", labelcolor="blue")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    handles = [line_micro, line_rel_micro, line_macro, line_rel_macro]
    ax.legend(handles, [handle.get_label() for handle in handles], fontsize="small", loc="best")
    ax.set_title(f"{title_prefix}Population Size")
    ax.grid(True, alpha=0.3)


def _plot_knowledgeCPU3(ax, mean_know, std_know, mean_generalization, boundary_one, boundary_two, title_prefix="", knowledge_supported: bool = True):
    knowledge = mean_know * 100
    deviation = std_know * 100
    generalization = mean_generalization * 100
    line_know, = ax.plot(knowledge, color="purple", label="Knowledge %")
    ax.fill_between(range(len(knowledge)), knowledge - deviation, knowledge + deviation, color="purple", alpha=0.15)
    line_gen, = ax.plot(generalization, color="orange", label="Generalization %")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title_prefix}Knowledge and Generalization")
    ax.legend([line_know, line_gen], [line_know.get_label(), line_gen.get_label()], fontsize="small", loc="best")
    ax.grid(True, alpha=0.3)
    if not knowledge_supported:
        ax.text(0.5, 0.9, "Knowledge unavailable for this environment type.", ha="center", va="center", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))


def _plot_reward_qualityCPU3(ax, mean_avg_r, mean_avg_rel_r, mean_avg_q_all, mean_avg_q_rel, boundary_one, boundary_two, title_prefix=""):
    line_r, = ax.plot(mean_avg_r, color="brown", label="R (All)")
    line_rel_r, = ax.plot(mean_avg_rel_r, color="green", label="R (Reliable)")
    twin = ax.twinx()
    line_q, = twin.plot(mean_avg_q_all, color="blue", label="Q (All)")
    line_rel_q, = twin.plot(mean_avg_q_rel, color="cyan", label="Q (Reliable)")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    twin.set_ylim(0, 1.05)
    handles = [line_r, line_rel_r, line_q, line_rel_q]
    ax.legend(handles, [handle.get_label() for handle in handles], fontsize="small")
    ax.set_title(f"{title_prefix}Reward and Quality")
    ax.grid(True, alpha=0.3)


def _plot_policy_mapCPU3(fig, ax, best_agent: ACS2CPU3, env: GridEnvironmentCPU3, policy_avg_steps: float, title_prefix=""):
    ax.set_title(f"{title_prefix}Policy Map | Avg Steps: {policy_avg_steps:.2f}")
    ax.set_xlim(0, env.cols)
    ax.set_ylim(env.rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=1000)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05))
    cbar.set_label("Fitness (Q*R)")
    width = 1 / 3.0
    offsets = {4: (0, 0), 2: (width, 0), 5: (2 * width, 0), 0: (0, width), 1: (2 * width, width), 6: (0, 2 * width), 3: (width, 2 * width), 7: (2 * width, 2 * width)}
    for row in range(env.rows):
        for col in range(env.cols):
            ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor="black", edgecolor="gray", linewidth=0.5))
            position = (row, col)
            if position in env.obstacles:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor="#444444", hatch="///"))
                continue
            if position == env.goal_pos:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor="gold"))
                ax.text(col + 0.5, row + 0.5, "G", ha="center", va="center", fontweight="bold", fontsize=14)
                continue
            state_bits = best_agent.state_to_bits([str(row), str(col)])
            match_set = [classifier for classifier in best_agent.population if classifier.matches_bits(state_bits)]
            action_fitness = {action: 0.0 for action in range(8)}
            for classifier in match_set:
                action_fitness[classifier.A] = max(classifier.fitness, action_fitness.get(classifier.A, -1.0))
            best_action = max(action_fitness, key=action_fitness.get) if action_fitness else -1
            for action_idx, (dx, dy) in offsets.items():
                fit = action_fitness.get(action_idx, 0.0)
                if fit > 0:
                    ax.add_patch(patches.Rectangle((col + dx, row + dy), width, width, facecolor=cmap(norm(fit)), linewidth=0))
            if best_action != -1 and action_fitness.get(best_action, 0.0) > 50:
                ax.text(col + 0.5, row + 0.5, ARROW_LABELS_CPU3.get(best_action, "?"), ha="center", va="center", color="black", fontsize=10, fontweight="bold", zorder=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="square,pad=0.02"))


def _plot_top_rulesCPU3(ax, best_agent: ACS2CPU3, title_prefix=""):
    ax.axis("off")
    ax.set_title(f"{title_prefix}Top 10 Rules")
    table_data = []
    top_classifiers = sorted(best_agent.population, key=lambda classifier: classifier.fitness, reverse=True)[:10]
    for classifier in top_classifiers:
        classifier.sync_from_bits()
        mark_str = ", ".join(["".join(sorted(mark_set)) for mark_set in classifier.M]) if classifier.M else ""
        table_data.append([classifier.C, ARROW_LABELS_CPU3.get(classifier.A, str(classifier.A)), classifier.E, mark_str, f"{classifier.q:.2f}", f"{classifier.r:.2f}", f"{classifier.fitness:.2f}", str(classifier.num)])
    if not table_data:
        table_data = [["-"] * 8]
    table = ax.table(cellText=table_data, colLabels=["C", "A", "E", "M", "Q", "R", "Fit", "Num"], loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)


def _plot_origin_distributionCPU3(ax, stats: Dict[str, Any], boundary_one: int, boundary_two: int, is_in_dashboard: bool = False, title_prefix: str = ""):
    episodes = np.arange(len(stats.get("mean_steps", np.array([])).flatten()))
    if len(episodes) == 0:
        ax.set_title(f"{title_prefix}Classifier Origin Distribution (No data)")
        return
    labels = ["Covering %", "GA %", "ALP Unexpected %", "ALP Expected %", "ALP Covering %"]
    keys = ["mean_covering_perc", "mean_ga_perc", "mean_alp_unexpected_perc", "mean_alp_expected_perc", "mean_alp_covering_perc"]
    colors = ["blue", "red", "green", "purple", "orange"]
    stacked = [stats.get(key, np.zeros_like(episodes)).flatten()[: len(episodes)] for key in keys]
    ax.stackplot(episodes, np.array(stacked), labels=labels, colors=colors, alpha=0.8)
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    ax.set_title(f"{title_prefix}Classifier Origin Distribution")
    ax.set_ylim(0, 100)
    if is_in_dashboard:
        ax.legend(fontsize="x-small", loc="best")
    else:
        ax.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)


def _plot_origin_distribution_absCPU3(ax, stats: Dict[str, Any], boundary_one: int, boundary_two: int, is_in_dashboard: bool = False, title_prefix: str = ""):
    episodes = np.arange(len(stats.get("mean_steps", np.array([])).flatten()))
    if len(episodes) == 0:
        ax.set_title(f"{title_prefix}Absolute Origin Distribution (No data)")
        return
    labels = ["Covering", "GA", "ALP Unexpected", "ALP Expected", "ALP Covering"]
    keys = ["mean_covering_abs", "mean_ga_abs", "mean_alp_unexpected_abs", "mean_alp_expected_abs", "mean_alp_covering_abs"]
    colors = ["blue", "red", "green", "purple", "orange"]
    for key, label, color in zip(keys, labels, colors):
        ax.plot(episodes, stats.get(key, np.zeros_like(episodes)).flatten()[: len(episodes)], label=label, color=color)
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    ax.set_title(f"{title_prefix}Absolute Origin Distribution")
    if is_in_dashboard:
        ax.legend(fontsize="x-small", loc="best")
    else:
        ax.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)


def plot_grouped_bar_chartCPU3(ax, stats: Dict[str, Any], origin_key: str, title: str, params_phases: Dict[str, Any], plot_absolute_annotations: bool = False, title_prefix: str = ""):
    mean_dist = stats.get(f"mean_{origin_key}_creation_dist", np.zeros((10, 10)))
    abs_dist = stats.get(f"mean_{origin_key}_creation_dist_abs", np.zeros((10, 10)))
    if not np.any(abs_dist):
        ax.set_title(f"{title_prefix}{title}")
        ax.text(0.5, 0.5, "No classifiers of this origin were created.", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_xlabel("Observation Episode")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        return [], []
    observation_count = max(1, mean_dist.shape[0])
    creation_count = max(1, mean_dist.shape[1])
    total_episodes = max(1, sum(phase["episodes"] for phase in params_phases.values()))
    interval_size = total_episodes / observation_count
    creation_interval_size = total_episodes / creation_count
    observation_labels = [f"Ep {int((i + 1) * interval_size)}" for i in range(observation_count)]
    creation_labels = [f"Ep {int(i * creation_interval_size) + 1}-{int((i + 1) * creation_interval_size)}" for i in range(creation_count)]
    cmap = plt.cm.get_cmap("plasma", creation_count)
    colors = [cmap(i) for i in range(creation_count)]
    bar_width = 0.8 / creation_count
    x = np.arange(observation_count)
    handles = []
    labels = []
    for idx in range(creation_count):
        container = ax.bar(x + idx * bar_width, mean_dist[:, idx], bar_width, label=creation_labels[idx], color=colors[idx])
        if container.patches:
            handles.append(container.patches[0])
            labels.append(creation_labels[idx])
        if plot_absolute_annotations:
            for row_idx, bar in enumerate(container):
                abs_value = abs_dist[row_idx, idx]
                if abs_value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(abs_value)}", ha="center", va="bottom", fontsize=7)
    ax.set_title(f"{title_prefix}{title}")
    ax.set_xlabel("Observation Episode")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x + (creation_count / 2 - 0.5) * bar_width)
    ax.set_xticklabels(observation_labels, rotation=45, ha="right", fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    return handles, labels


def create_loaded_dashboardCPU3(loaded_stats: Dict[str, Any], optimal_avg_steps: float, params_phases: Dict[str, Any], n_exp: int, n_steps: int, environment_metadata: Dict[str, Any], plot_steps: bool = False, plot_population: bool = False, plot_knowledge: bool = False, plot_reward_quality: bool = False, plot_policy_map: bool = False, plot_top_rules: bool = False, plot_origin_distribution: bool = False, plot_origin_distribution_abs: bool = False, plot_creation_dist_key: Optional[str] = None, plot_all_dashboards: bool = False, timestamp: Optional[str] = None, title_prefix: str = ""):
    title_prefix = _sanitize_title_prefixCPU3(title_prefix)
    env = environment_from_metadataCPU3(environment_metadata)
    active_plot_count = sum([plot_steps, plot_population, plot_knowledge, plot_reward_quality, plot_policy_map, plot_top_rules, plot_origin_distribution, plot_origin_distribution_abs, plot_creation_dist_key is not None])
    if not plot_all_dashboards and active_plot_count == 1:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        boundary_one = params_phases["explore"]["episodes"]
        boundary_two = boundary_one + params_phases["exploit1"]["episodes"]
        best_agent = loaded_stats.get("best_agent")
        plot_name = "unspecified"
        if plot_steps:
            mean_steps = loaded_stats.get("mean_steps", np.array([])).flatten()
            _plot_stepsCPU3(ax, mean_steps, calculate_emaCPU3(mean_steps, 50), optimal_avg_steps, loaded_stats.get("Exploit Avg. Steps", loaded_stats.get("Last Avg. Steps", 0)), boundary_one, boundary_two, 50, title_prefix)
            plot_name = "steps"
        elif plot_population:
            _plot_populationCPU3(ax, loaded_stats.get("mean_micro_pop", np.array([])).flatten(), loaded_stats.get("mean_rel_micro_pop", np.array([])).flatten(), loaded_stats.get("mean_macro_pop", np.array([])).flatten(), loaded_stats.get("mean_rel_macro_pop", np.array([])).flatten(), boundary_one, boundary_two, title_prefix)
            plot_name = "population"
        elif plot_knowledge:
            _plot_knowledgeCPU3(ax, loaded_stats.get("mean_know", np.array([])).flatten(), loaded_stats.get("std_know", np.array([])).flatten(), loaded_stats.get("mean_generalization", np.array([])).flatten(), boundary_one, boundary_two, title_prefix, env.supports_metric_evaluation)
            plot_name = "knowledge"
        elif plot_reward_quality:
            _plot_reward_qualityCPU3(ax, loaded_stats.get("mean_avg_r", np.array([])).flatten(), loaded_stats.get("mean_avg_rel_r", np.array([])).flatten(), loaded_stats.get("mean_avg_q_all", np.array([])).flatten(), loaded_stats.get("mean_avg_q_rel", np.array([])).flatten(), boundary_one, boundary_two, title_prefix)
            plot_name = "reward_quality"
        elif plot_policy_map and best_agent and env.supports_policy_map:
            _plot_policy_mapCPU3(fig, ax, best_agent, env, calculate_policy_avg_lenCPU3(best_agent, env), title_prefix)
            plot_name = "policy_map"
        elif plot_top_rules and best_agent:
            _plot_top_rulesCPU3(ax, best_agent, title_prefix)
            plot_name = "top_rules"
        elif plot_origin_distribution:
            _plot_origin_distributionCPU3(ax, loaded_stats, boundary_one, boundary_two, False, title_prefix)
            plot_name = "origin_distribution"
        elif plot_origin_distribution_abs:
            _plot_origin_distribution_absCPU3(ax, loaded_stats, boundary_one, boundary_two, False, title_prefix)
            plot_name = "origin_distribution_abs"
        elif plot_creation_dist_key:
            handles, labels = plot_grouped_bar_chartCPU3(ax, loaded_stats, plot_creation_dist_key, f"{plot_creation_dist_key.upper()} Creation Distribution", params_phases, True, title_prefix)
            ax.legend(handles, labels, title="Creation Interval", fontsize="x-small", loc="best")
            plot_name = f"creation_dist_{plot_creation_dist_key}"
        plt.tight_layout()
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./reports/acs2_project_dashboard_{timestamp}_{plot_name}.png")
        plt.close("all")
        return
    create_dashboardCPU3(loaded_stats.get("best_agent"), optimal_avg_steps, env, loaded_stats, params_phases, n_exp, n_steps, loaded_stats, plot_steps, plot_population, plot_knowledge, plot_reward_quality, plot_policy_map, plot_top_rules, plot_origin_distribution, plot_origin_distribution_abs, plot_creation_dist_key, plot_all_dashboards, timestamp, title_prefix)


def create_dashboardCPU3(best_agent: Optional[ACS2CPU3], optimal_avg_steps: float, env: Optional[EnvironmentCPU3], stats: Dict[str, Any], params_phases: Dict[str, Any], n_exp: int, n_steps: int, summary_stats: Dict[str, Any], plot_steps: bool = True, plot_population: bool = True, plot_knowledge: bool = True, plot_reward_quality: bool = True, plot_policy_map: bool = True, plot_top_rules: bool = True, plot_origin_distribution: bool = True, plot_origin_distribution_abs: bool = False, plot_creation_dist_key: Optional[str] = None, plot_all_dashboards: bool = False, timestamp: Optional[str] = None, title_prefix: str = ""):
    title_prefix = _sanitize_title_prefixCPU3(title_prefix)
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f"{title_prefix}ACS2 Dashboard: Final Report", fontsize=16, fontweight="bold")
    gs = GridSpec(10, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8])
    boundary_one = params_phases["explore"]["episodes"]
    boundary_two = boundary_one + params_phases["exploit1"]["episodes"]
    if plot_all_dashboards or plot_steps:
        mean_steps = stats.get("mean_steps", np.array([])).flatten()
        _plot_stepsCPU3(fig.add_subplot(gs[0, 0]), mean_steps, calculate_emaCPU3(mean_steps, 50), optimal_avg_steps, summary_stats.get("Exploit Avg. Steps", summary_stats.get("Last Avg. Steps", 0)), boundary_one, boundary_two, 50, title_prefix)
    if plot_all_dashboards or plot_population:
        _plot_populationCPU3(fig.add_subplot(gs[0, 1]), stats.get("mean_micro_pop", np.array([])).flatten(), stats.get("mean_rel_micro_pop", np.array([])).flatten(), stats.get("mean_macro_pop", np.array([])).flatten(), stats.get("mean_rel_macro_pop", np.array([])).flatten(), boundary_one, boundary_two, title_prefix)
    if plot_all_dashboards or plot_knowledge:
        _plot_knowledgeCPU3(fig.add_subplot(gs[0, 2]), stats.get("mean_know", np.array([])).flatten(), stats.get("std_know", np.array([])).flatten(), stats.get("mean_generalization", np.array([])).flatten(), boundary_one, boundary_two, title_prefix, env.supports_metric_evaluation if env else True)
    if plot_all_dashboards or plot_reward_quality:
        _plot_reward_qualityCPU3(fig.add_subplot(gs[0, 3]), stats.get("mean_avg_r", np.array([])).flatten(), stats.get("mean_avg_rel_r", np.array([])).flatten(), stats.get("mean_avg_q_all", np.array([])).flatten(), stats.get("mean_avg_q_rel", np.array([])).flatten(), boundary_one, boundary_two, title_prefix)
    if (plot_all_dashboards or plot_policy_map) and best_agent and env and env.supports_policy_map:
        _plot_policy_mapCPU3(fig, fig.add_subplot(gs[1, 0:2]), best_agent, env, calculate_policy_avg_lenCPU3(best_agent, env), title_prefix)
    if (plot_all_dashboards or plot_top_rules) and best_agent:
        top_rules_slot = gs[1, 2:4] if env and env.supports_policy_map else gs[1, :]
        _plot_top_rulesCPU3(fig.add_subplot(top_rules_slot), best_agent, title_prefix)
    if plot_all_dashboards or plot_origin_distribution:
        _plot_origin_distributionCPU3(fig.add_subplot(gs[2, 0:2]), stats, boundary_one, boundary_two, True, title_prefix)
    if plot_all_dashboards or plot_origin_distribution_abs:
        _plot_origin_distribution_absCPU3(fig.add_subplot(gs[2, 2:4]), stats, boundary_one, boundary_two, True, title_prefix)

    creation_plot_keys = ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]
    grouped_handles = []
    grouped_labels = []
    if plot_all_dashboards or plot_creation_dist_key in creation_plot_keys:
        for idx, key in enumerate(creation_plot_keys):
            if plot_all_dashboards or key == plot_creation_dist_key:
                ax = fig.add_subplot(gs[3 + idx, :])
                handles, labels = plot_grouped_bar_chartCPU3(ax, stats, key, f"{key.upper()} Creation Distribution", params_phases, plot_all_dashboards, title_prefix)
                if not grouped_handles:
                    grouped_handles = handles
                    grouped_labels = labels
                if not plot_all_dashboards or idx < len(creation_plot_keys) - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
    if plot_all_dashboards and grouped_handles:
        legend_ax = fig.add_subplot(gs[8, :])
        legend_ax.legend(grouped_handles, grouped_labels, title="Creation Interval", ncol=5, loc="center")
        legend_ax.axis("off")

    report_ax = fig.add_subplot(gs[9, :])
    report_ax.axis("off")
    if best_agent and env:
        exploit_avg_std = calculate_exploit_avg_stdCPU3(stats, params_phases)
        best_policy = calculate_policy_avg_lenCPU3(best_agent, env) if env.supports_policy_map else 0.0
        knowledge_text = f"{summary_stats.get('Knowledge', 0) * 100:.2f}%" if env.supports_metric_evaluation else "N/A (unsupported)"
        lines = [
            "--- PARAMETER & RESULT SUMMARY ---",
            "=" * 110,
            f"[ENV]     Type: {env.type} | Name: {env.name}",
            f"[GENERAL] N_EXP: {n_exp} | MAX_STEPS: {n_steps} | u_max: {best_agent.cfg.u_max} | Theta_GA: {best_agent.cfg.theta_ga}",
            f"          Gamma: {best_agent.cfg.gamma} | Beta: {best_agent.cfg.beta} | Theta_i: {best_agent.cfg.theta_i} | Theta_r: {best_agent.cfg.theta_r} | Theta_AS: {best_agent.cfg.theta_as}",
            f"[ALP]     Marking: {_alp_marking_labelCPU3(summary_stats.get('ALP Marking', getattr(best_agent.cfg, 'alp_mark_only_incorrect', True)))}",
            f"[SUBSUMPTION] Status: {summary_stats.get('Subsumption', 'Unknown')}",
            f"[GA]      Status: {'ENABLED' if any(phase.get('ga', False) for phase in params_phases.values()) else 'DISABLED'}",
            f"[TIME]    Total: {summary_stats.get('Total Time', 0.0):.2f}s | Avg: {summary_stats.get('Avg Time', 0.0):.2f}s +/- {summary_stats.get('Std Time', 0.0):.2f}s",
            "-" * 110,
            f"{'PHASE':<12} | {'EPISODES':<10} | {'EPSILON':<10} | {'BETA':<10} | {'ALP':<6} | {'GA':<6} | {'DECAY':<6}",
        ]
        for name in ("explore", "exploit1", "exploit2"):
            phase = params_phases.get(name)
            if phase:
                lines.append(f"{name:<12} | {phase['episodes']:<10} | {phase['epsilon']:<10} | {phase['beta']:<10} | {str(phase.get('alp', False)):<6} | {str(phase.get('ga', False)):<6} | {str(phase.get('decay', False)):<6}")
        lines.extend([
            "-" * 110,
            "[RESULTS - AVERAGES FROM EXPLOIT2]",
            f" >> Steps (Exploit Avg.): {summary_stats.get('Exploit Avg. Steps', summary_stats.get('Last Avg. Steps', 0)):.2f} +/- {exploit_avg_std:.2f} vs Optimal: {optimal_avg_steps:.2f} vs Best Policy: {best_policy:.2f}",
            f" >> Knowledge: {knowledge_text} | Generalization: {summary_stats.get('Generalization', 0) * 100:.2f}%",
            f" >> Avg Reward (All): {summary_stats.get('Rew (All)', 0):.2f} | Avg Reward (Rel): {summary_stats.get('Rew (Rel)', 0):.2f}",
            f" >> Population Micro: {summary_stats.get('Micro', 0):.1f} (Rel: {summary_stats.get('Micro (Rel)', 0):.1f}) | Macro: {summary_stats.get('Macro', 0):.1f} (Rel: {summary_stats.get('Macro (Rel)', 0):.1f})",
        ])
        report_ax.text(0.01, 1.0, "\n".join(lines), transform=report_ax.transAxes, fontsize=9, fontfamily="monospace", va="top")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_suffix = "all" if plot_all_dashboards else '_'.join(plot_names) if plot_names else "unspecified"
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./reports/acs2_project_dashboard_{timestamp}_{plot_suffix}.png")
    plt.close("all")
