from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from environment.runtime_gpu4 import EnvironmentGPU4, GridEnvironmentGPU4, environment_from_metadataGPU4
from src.models.acs2.acs2GPU4 import AgentSelectionGPU4


ARROW_LABELS_GPU4 = {0: "\u2190", 1: "\u2192", 2: "\u2191", 3: "\u2193", 4: "\u2196", 5: "\u2197", 6: "\u2199", 7: "\u2198"}
CREATION_KEYS_GPU4 = ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]


def _alp_marking_labelGPU4(value: Any) -> str:
    if value in (True, "incorrect_only", "Restricted (on incorrect)"):
        return "Restricted (on incorrect)"
    if value in (False, "all_action_set", "Full Action Set"):
        return "Full Action Set"
    return str(value)


def calculate_emaGPU4(data: np.ndarray, span: int) -> np.ndarray:
    if len(data) == 0:
        return np.array([])
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for idx in range(1, len(data)):
        ema[idx] = alpha * data[idx] + (1 - alpha) * ema[idx - 1]
    return ema


def _action_fitness_mapGPU4(best_agent: AgentSelectionGPU4, states: torch.Tensor) -> torch.Tensor:
    return best_agent.agent.action_fitness_for_states(best_agent.experiment_index, states)


def calculate_policy_avg_lenGPU4(best_agent: Optional[AgentSelectionGPU4], environment: EnvironmentGPU4) -> float:
    if best_agent is None or not isinstance(environment, GridEnvironmentGPU4):
        return 0.0
    valid_states = environment.valid_coords
    solved_steps = []
    for start_idx in range(valid_states.shape[0]):
        current = valid_states[start_idx : start_idx + 1].clone()
        visited = set()
        for step in range(1, 51):
            key = tuple(current[0].tolist())
            if key in visited:
                break
            visited.add(key)
            action_fitness = _action_fitness_mapGPU4(best_agent, current)
            if torch.all(action_fitness <= -1e8):
                break
            best_action = int(torch.argmax(action_fitness[0]).item())
            current = environment.peek_step(current, torch.tensor([best_action], device=environment.device))
            if torch.all(current[0] == environment.goal_pos):
                solved_steps.append(step)
                break
    return float(np.mean(solved_steps)) if solved_steps else 0.0


def calculate_exploit_avg_stdGPU4(stats: Dict[str, Any], params_phases: Dict[str, Any]) -> float:
    steps = np.asarray(stats.get("stats_steps", np.array([])), dtype=float)
    if steps.size == 0:
        return 0.0
    exploit2_start = int(params_phases.get("explore", {}).get("episodes", 0)) + int(params_phases.get("exploit1", {}).get("episodes", 0))
    exploit2_slice = steps[:, exploit2_start:] if steps.ndim == 2 else np.array([])
    if exploit2_slice.size == 0:
        exploit2_slice = steps
    return float(np.std(exploit2_slice))


def _plot_stepsGPU4(ax, mean_steps, ema_values, optimal_avg_steps, exploit_avg, boundary_one, boundary_two, ema_span, title_prefix=""):
    raw_line, = ax.plot(mean_steps, color="lightblue", label=f"Mean Steps (Exploit Avg.: {exploit_avg:.2f})")
    ema_line, = ax.plot(ema_values, color="red", linestyle="--", label=f"EMA {ema_span}")
    optimal_line = ax.axhline(y=optimal_avg_steps, color="magenta", linestyle="-.", label=f"Optimal: {optimal_avg_steps:.2f}")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    ax.set_title(f"{title_prefix}Steps to Goal")
    ax.legend(handles=[ema_line, raw_line, optimal_line], fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_populationGPU4(ax, mean_micro, mean_rel_micro, mean_macro, mean_rel_macro, boundary_one, boundary_two, title_prefix=""):
    line_micro, = ax.plot(mean_micro, color="green", label="Micro (All)")
    line_rel_micro, = ax.plot(mean_rel_micro, color="green", linestyle="--", label="Micro (Rel)")
    twin = ax.twinx()
    line_macro, = twin.plot(mean_macro, color="blue", label="Macro (All)")
    line_rel_macro, = twin.plot(mean_rel_macro, color="blue", linestyle="--", label="Macro (Rel)")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    handles = [line_micro, line_rel_micro, line_macro, line_rel_macro]
    ax.legend(handles, [handle.get_label() for handle in handles], fontsize="small")
    ax.set_title(f"{title_prefix}Population Size")
    ax.grid(True, alpha=0.3)


def _plot_knowledgeGPU4(ax, mean_know, std_know, mean_generalization, boundary_one, boundary_two, title_prefix="", knowledge_supported: bool = True):
    knowledge = mean_know * 100
    std = std_know * 100
    generalization = mean_generalization * 100
    line_know, = ax.plot(knowledge, color="purple", label="Knowledge %")
    ax.fill_between(range(len(knowledge)), knowledge - std, knowledge + std, color="purple", alpha=0.15)
    line_gen, = ax.plot(generalization, color="orange", label="Generalization %")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    ax.legend([line_know, line_gen], [line_know.get_label(), line_gen.get_label()], fontsize="small")
    ax.set_ylim(0, 105)
    ax.set_title(f"{title_prefix}Knowledge and Generalization")
    ax.grid(True, alpha=0.3)
    if not knowledge_supported:
        ax.text(0.5, 0.9, "Knowledge unavailable for this environment type.", ha="center", va="center", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))


def _plot_reward_qualityGPU4(ax, mean_avg_r, mean_avg_rel_r, mean_avg_q_all, mean_avg_q_rel, boundary_one, boundary_two, title_prefix=""):
    line_r, = ax.plot(mean_avg_r, color="brown", label="R (All)")
    line_rr, = ax.plot(mean_avg_rel_r, color="green", label="R (Reliable)")
    twin = ax.twinx()
    line_q, = twin.plot(mean_avg_q_all, color="blue", label="Q (All)")
    line_qr, = twin.plot(mean_avg_q_rel, color="cyan", label="Q (Reliable)")
    handles = [line_r, line_rr, line_q, line_qr]
    ax.legend(handles, [handle.get_label() for handle in handles], fontsize="small")
    ax.axvline(x=boundary_one, color="orange", linestyle="--")
    ax.axvline(x=boundary_two, color="orange", linestyle="--")
    twin.set_ylim(0, 1.05)
    ax.set_title(f"{title_prefix}Reward and Quality")
    ax.grid(True, alpha=0.3)


def _plot_policy_mapGPU4(fig, ax, best_agent: AgentSelectionGPU4, environment: GridEnvironmentGPU4, policy_avg_steps: float, title_prefix=""):
    ax.set_title(f"{title_prefix}Policy Map | Avg Steps: {policy_avg_steps:.2f}")
    ax.set_xlim(0, environment.cols)
    ax.set_ylim(environment.rows, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=1000)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05))
    states = torch.stack([torch.tensor([row, col], device=environment.device) for row in range(environment.rows) for col in range(environment.cols)], dim=0)
    action_fitness = _action_fitness_mapGPU4(best_agent, states).detach().cpu().numpy()
    width = 1 / 3.0
    offsets = {4: (0, 0), 2: (width, 0), 5: (2 * width, 0), 0: (0, width), 1: (2 * width, width), 6: (0, 2 * width), 3: (width, 2 * width), 7: (2 * width, 2 * width)}
    metadata = environment.to_metadata()
    obstacle_set = {tuple(item) for item in metadata["parameters"]["obstacles"]}
    goal = tuple(environment.goal_pos.cpu().tolist())
    for row in range(environment.rows):
        for col in range(environment.cols):
            idx = row * environment.cols + col
            ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor="black", edgecolor="gray", linewidth=0.5))
            if (row, col) in obstacle_set:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor="#444444", hatch="///"))
                continue
            if (row, col) == goal:
                ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor="gold"))
                ax.text(col + 0.5, row + 0.5, "G", ha="center", va="center", fontweight="bold", fontsize=14)
                continue
            best_action = int(np.argmax(action_fitness[idx]))
            for action_idx, (dx, dy) in offsets.items():
                fit = action_fitness[idx, action_idx]
                if fit > 0:
                    ax.add_patch(patches.Rectangle((col + dx, row + dy), width, width, facecolor=cmap(norm(fit)), linewidth=0))
            if action_fitness[idx, best_action] > 50:
                ax.text(col + 0.5, row + 0.5, ARROW_LABELS_GPU4.get(best_action, "?"), ha="center", va="center", color="black", fontsize=12, fontweight="bold", zorder=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="square,pad=0.02"))


def _format_rule_partGPU4(values: np.ndarray) -> str:
    rendered = ["#" if int(value) < 0 else str(int(value)) for value in values.tolist()]
    return "[" + " ".join(rendered) + "]"


def _format_marksGPU4(values: np.ndarray) -> str:
    columns = []
    for feature_marks in values:
        indices = np.flatnonzero(feature_marks)
        columns.append("".join(str(int(idx)) for idx in indices) if indices.size else "-")
    return " | ".join(columns)


def _top_rulesGPU4(best_agent: AgentSelectionGPU4) -> List[List[str]]:
    exp_idx = best_agent.experiment_index
    agent = best_agent.agent
    active_idx = torch.where(agent.active_mask[exp_idx])[0]
    if len(active_idx) == 0:
        return []
    fitness = (agent.q[exp_idx] * agent.r[exp_idx])[active_idx]
    order = torch.argsort(fitness, descending=True)[:10]
    selected_idx = active_idx[order]
    cond = agent.C[exp_idx, selected_idx].detach().cpu().numpy()
    action = agent.A[exp_idx, selected_idx].detach().cpu().numpy()
    effect = agent.E[exp_idx, selected_idx].detach().cpu().numpy()
    marks = agent.M[exp_idx, selected_idx].detach().cpu().numpy()
    q_vals = agent.q[exp_idx, selected_idx].detach().cpu().numpy()
    r_vals = agent.r[exp_idx, selected_idx].detach().cpu().numpy()
    nums = agent.num[exp_idx, selected_idx].detach().cpu().numpy()
    rows: List[List[str]] = []
    for idx in range(len(selected_idx)):
        action_label = ARROW_LABELS_GPU4.get(int(action[idx]), str(int(action[idx])))
        rows.append([
            _format_rule_partGPU4(cond[idx]),
            action_label,
            _format_rule_partGPU4(effect[idx]),
            _format_marksGPU4(marks[idx]),
            f"{float(q_vals[idx]):.2f}",
            f"{float(r_vals[idx]):.2f}",
            f"{float(q_vals[idx] * r_vals[idx]):.2f}",
            str(int(nums[idx])),
        ])
    return rows


def _plot_top_rulesGPU4(ax, best_agent: Optional[AgentSelectionGPU4], title_prefix=""):
    ax.axis("off")
    ax.set_title(f"{title_prefix}Top 10 Rules")
    table_data = _top_rulesGPU4(best_agent) if best_agent is not None else []
    if not table_data:
        table_data = [["-"] * 8]
    table = ax.table(cellText=table_data, colLabels=["C", "A", "E", "M", "Q", "R", "Fit", "Num"], loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)


def _plot_origin_distributionGPU4(ax, stats: Dict[str, Any], boundary_one: int, boundary_two: int, is_in_dashboard: bool = False, title_prefix: str = ""):
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


def _plot_origin_distribution_absGPU4(ax, stats: Dict[str, Any], boundary_one: int, boundary_two: int, is_in_dashboard: bool = False, title_prefix: str = ""):
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


def plot_grouped_bar_chartGPU4(ax, stats: Dict[str, Any], origin_key: str, title: str, params_phases: Dict[str, Any], plot_absolute_annotations: bool = False, title_prefix: str = "") -> Tuple[List[Any], List[str]]:
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
    handles: List[Any] = []
    labels: List[str] = []
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


def create_dashboardGPU4(best_agent: Optional[AgentSelectionGPU4], optimal_avg_steps: float, environment: EnvironmentGPU4, stats: Dict[str, Any], params_phases: Dict[str, Any], n_exp: int, n_steps: int, summary_stats: Dict[str, Any], plot_all_dashboards: bool = True, timestamp: Optional[str] = None, title_prefix: str = ""):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f"{title_prefix}ACS2 Dashboard: Final Report", fontsize=16, fontweight="bold")
    gs = GridSpec(10, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8])
    boundary_one = params_phases["explore"]["episodes"]
    boundary_two = boundary_one + params_phases["exploit1"]["episodes"]
    mean_steps = stats.get("mean_steps", np.array([])).flatten()
    _plot_stepsGPU4(fig.add_subplot(gs[0, 0]), mean_steps, calculate_emaGPU4(mean_steps, 50), optimal_avg_steps, summary_stats.get("Exploit Avg. Steps", summary_stats.get("Last Avg. Steps", 0)), boundary_one, boundary_two, 50, title_prefix)
    _plot_populationGPU4(fig.add_subplot(gs[0, 1]), stats.get("mean_micro_pop", np.array([])).flatten(), stats.get("mean_rel_micro_pop", np.array([])).flatten(), stats.get("mean_macro_pop", np.array([])).flatten(), stats.get("mean_rel_macro_pop", np.array([])).flatten(), boundary_one, boundary_two, title_prefix)
    _plot_knowledgeGPU4(fig.add_subplot(gs[0, 2]), stats.get("mean_know", np.array([])).flatten(), stats.get("std_know", np.array([])).flatten(), stats.get("mean_generalization", np.array([])).flatten(), boundary_one, boundary_two, title_prefix, environment.supports_metric_evaluation)
    _plot_reward_qualityGPU4(fig.add_subplot(gs[0, 3]), stats.get("mean_avg_r", np.array([])).flatten(), stats.get("mean_avg_rel_r", np.array([])).flatten(), stats.get("mean_avg_q_all", np.array([])).flatten(), stats.get("mean_avg_q_rel", np.array([])).flatten(), boundary_one, boundary_two, title_prefix)
    if best_agent is not None and environment.supports_policy_map:
        _plot_policy_mapGPU4(fig, fig.add_subplot(gs[1, 0:2]), best_agent, environment, calculate_policy_avg_lenGPU4(best_agent, environment), title_prefix)
        _plot_top_rulesGPU4(fig.add_subplot(gs[1, 2:4]), best_agent, title_prefix)
    else:
        _plot_top_rulesGPU4(fig.add_subplot(gs[1, :]), best_agent, title_prefix)
    _plot_origin_distributionGPU4(fig.add_subplot(gs[2, 0:2]), stats, boundary_one, boundary_two, True, title_prefix)
    _plot_origin_distribution_absGPU4(fig.add_subplot(gs[2, 2:4]), stats, boundary_one, boundary_two, True, title_prefix)

    grouped_handles: List[Any] = []
    grouped_labels: List[str] = []
    for idx, key in enumerate(CREATION_KEYS_GPU4):
        ax = fig.add_subplot(gs[3 + idx, :])
        handles, labels = plot_grouped_bar_chartGPU4(ax, stats, key, f"{key.upper()} Creation Distribution", params_phases, True, title_prefix)
        if not grouped_handles and handles:
            grouped_handles = handles
            grouped_labels = labels
        if idx < len(CREATION_KEYS_GPU4) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    if grouped_handles:
        legend_ax = fig.add_subplot(gs[8, :])
        legend_ax.legend(grouped_handles, grouped_labels, title="Creation Interval", ncol=5, loc="center")
        legend_ax.axis("off")

    report_ax = fig.add_subplot(gs[9, :])
    report_ax.axis("off")
    cfg = best_agent.agent.cfg if best_agent is not None else None
    exploit_avg_std = calculate_exploit_avg_stdGPU4(stats, params_phases)
    best_policy = calculate_policy_avg_lenGPU4(best_agent, environment) if best_agent is not None and environment.supports_policy_map else 0.0
    knowledge_text = f"{summary_stats.get('Knowledge', 0) * 100:.2f}%" if environment.supports_metric_evaluation else "N/A (unsupported)"
    lines = [
        "--- PARAMETER & RESULT SUMMARY ---",
        "=" * 110,
        f"[ENV]     Type: {environment.type} | Name: {environment.name}",
        f"[GENERAL] N_EXP: {n_exp} | MAX_STEPS: {n_steps} | u_max: {cfg.u_max if cfg else '-'} | Theta_GA: {cfg.theta_ga if cfg else '-'}",
        f"          Gamma: {cfg.gamma if cfg else '-'} | Beta: {cfg.beta if cfg else '-'} | Theta_i: {cfg.theta_i if cfg else '-'} | Theta_r: {cfg.theta_r if cfg else '-'} | Theta_AS: {cfg.theta_as if cfg else '-'}",
        f"[ALP]     Marking: {_alp_marking_labelGPU4(summary_stats.get('ALP Marking', getattr(cfg, 'alp_mark_only_incorrect', True) if cfg else '-'))}",
        f"[GA]      Status: {'ENABLED' if any(phase.get('ga', False) for phase in params_phases.values()) else 'DISABLED'}",
        f"[TIME]    Total: {summary_stats.get('Total Time', 0.0):.2f}s | Avg: {summary_stats.get('Avg Time', 0.0):.2f}s | Std: {summary_stats.get('Std Time', 0.0):.2f}s",
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
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./reports/acs2_project_dashboard_{timestamp}_all.png")
    plt.close("all")


def create_loaded_dashboardGPU4(loaded_stats: Dict[str, Any], optimal_avg_steps: float, params_phases: Dict[str, Any], n_exp: int, n_steps: int, environment_metadata: Dict[str, Any], timestamp: Optional[str] = None, title_prefix: str = ""):
    environment = environment_from_metadataGPU4(environment_metadata, n_exp=1, device="cpu")
    create_dashboardGPU4(None, optimal_avg_steps, environment, loaded_stats, params_phases, n_exp, n_steps, loaded_stats, timestamp=timestamp, title_prefix=title_prefix)
