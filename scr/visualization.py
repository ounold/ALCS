import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, Any, Optional

from .models.acs2.acs2 import ACS2
from .models.acs2.classifier import Classifier
from .environment import GridEnvironment


def _get_lineage_str(classifier: Classifier, pop_map: Dict[int, Classifier], depth=2) -> str:
    if depth < 0 or classifier.parents is None:
        return ""
    p1_id, p2_id = classifier.parents
    p1, p2 = pop_map.get(p1_id), pop_map.get(p2_id)
    indent = " " * (4 * (2 - depth))
    lineage = f"\n{indent}├── Parent {p1_id}" if p1 else f"\n{indent}├── Parent {p1_id} (not in pop)"
    if p1:
        lineage += _get_lineage_str(p1, pop_map, depth - 1)
    lineage += f"\n{indent}└── Parent {p2_id}" if p2 else f"\n{indent}└── Parent {p2_id} (not in pop)"
    if p2:
        lineage += _get_lineage_str(p2, pop_map, depth - 1)
    return lineage


def calculate_ema(data, span):
    if len(data) == 0:
        return np.array([])
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data, dtype=float)
    if len(data) > 0:
        ema[0] = data[0]
        for t in range(1, len(data)):
            ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

def calculate_policy_avg_len(agent: Optional[ACS2], env: GridEnvironment) -> float:
    if not agent:
        return 0.0
    total_steps, solved_cases, max_sim_steps = 0, 0, 50
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) == env.goal_pos or (r, c) in env.obstacles:
                continue
            curr, steps, path_found = [r, c], 0, False
            for _ in range(max_sim_steps):
                state_str = [str(curr[0]), str(curr[1])]
                match_set = [cl for cl in agent.population if cl.matches(state_str)]
                if not match_set:
                    break
                act_fit = {}
                for cl in match_set:
                    if cl.fitness > act_fit.get(cl.A, -1.0):
                        act_fit[cl.A] = cl.fitness
                if not act_fit:
                    break
                best_a = max(act_fit, key=act_fit.get)
                curr = [int(n) for n in env.peek_step(curr, best_a)]
                steps += 1
                if tuple(curr) == env.goal_pos:
                    path_found = True
                    break
            if path_found:
                total_steps += steps
                solved_cases += 1
    return total_steps / solved_cases if solved_cases > 0 else 0.0


def _plot_steps(ax, mean_steps, ema_values, optimal_avg_steps, exploit_avg_val, b1, b2, ema_span, title_prefix: str = ""):
    l1, = ax.plot(mean_steps, color='lightblue', label=f'Mean Steps (Exploit Avg.: {exploit_avg_val:.2f})')
    l2, = ax.plot(ema_values, color='red', linestyle='--', label=f'EMA {ema_span}')
    l3 = ax.axhline(y=optimal_avg_steps, color='magenta', linestyle='-.', label=f'Optimal: {optimal_avg_steps:.2f}')
    ax.axvline(x=b1, color='orange', linestyle='--'), ax.axvline(x=b2, color='orange', linestyle='--')
    ax.set_title(f"{title_prefix}Steps to Goal"), ax.legend(handles=[l2, l1, l3], fontsize='small'), ax.grid(True, alpha=0.3)

def _plot_population(ax, mean_pop_micro, mean_rel_micro, mean_pop_macro, mean_rel_macro, b1, b2, title_prefix: str = ""):
    l_micro1, = ax.plot(mean_pop_micro, color='green', label='Micro (All)')
    l_micro2, = ax.plot(mean_rel_micro, color='green', linestyle='--', label='Micro (Rel)', alpha=0.8)
    ax.set_ylabel('Micro-population Size', color='green'), ax.tick_params(axis='y', labelcolor='green')
    ax2b = ax.twinx()
    l_macro1, = ax2b.plot(mean_pop_macro, color='blue', label='Macro (All)')
    l_macro2, = ax2b.plot(mean_rel_macro, color='blue', linestyle='--', label='Macro (Rel)')
    ax2b.set_ylabel('Macro-population Size (Unique Rules)', color='blue'), ax2b.tick_params(axis='y', labelcolor='blue')
    ax.axvline(x=b1, color='orange', linestyle='--'), ax.axvline(x=b2, color='orange', linestyle='--')
    lns = [l_micro1, l_micro2, l_macro1, l_macro2]
    ax.legend(lns, [l.get_label() for l in lns], fontsize='small', loc='best')
    ax.set_title(f"{title_prefix}Population Size (Dual Scale)"), ax.grid(True, alpha=0.3)

def _plot_knowledge(ax, mean_know, std_know, mean_generalization, b1, b2, title_prefix: str = ""):
    mk, sk, mg = mean_know * 100, std_know * 100, mean_generalization * 100
    p_know, = ax.plot(mk, color='purple', label='Knowledge %')
    ax.fill_between(range(len(mk)), mk - sk, mk + sk, color='purple', alpha=0.15)
    p_gen, = ax.plot(mg, color='orange', label='Generalization %')
    ax.set_ylabel('Percentage (%)'), ax.tick_params(axis='y'), ax.set_ylim(0, 105)
    ax.axvline(x=b1, color='orange', linestyle='--'), ax.axvline(x=b2, color='orange', linestyle='--')
    ax.set_title(f"{title_prefix}Knowledge & Generalization")
    ax.legend([p_know, p_gen], [l.get_label() for l in [p_know, p_gen]], fontsize='small', loc='best')
    ax.grid(True, alpha=0.3)

def _plot_reward_quality(ax, mean_avg_r, mean_avg_rel_r, mean_avg_q_all, mean_avg_q_rel, b1, b2, title_prefix: str = ""):
    p1, = ax.plot(mean_avg_r, color='brown', label='R (All)')
    p2, = ax.plot(mean_avg_rel_r, color='green', linestyle='-', linewidth=1.5, label='R (Reliable)')
    ax.set_ylabel('Reward')
    ax_q = ax.twinx()
    p5, = ax_q.plot(mean_avg_q_all, color='blue', label='Q (All)')
    p6, = ax_q.plot(mean_avg_q_rel, color='cyan', linestyle='-', linewidth=1.5, label='Q (Reliable)')
    ax_q.set_ylabel('Quality', color='blue'), ax_q.tick_params(axis='y', labelcolor='blue'), ax_q.set_ylim(0, 1.05)
    ax.axvline(x=b1, color='orange', linestyle='--'), ax.axvline(x=b2, color='orange', linestyle='--')
    ax.set_title(f"{title_prefix}Reward & Quality")
    lns = [p1, p2, p5, p6]
    ax.legend(lns, [l.get_label() for l in lns], fontsize='small'), ax.grid(True, alpha=0.3)

def _plot_policy_map(fig, ax, best_agent, env, policy_avg_steps, title_prefix: str = ""):
    ax.set_title(f"{title_prefix}Policy Map (Best Agent) | Avg Steps: {policy_avg_steps:.2f}")
    ax.set_xlim(0, env.cols), ax.set_ylim(env.rows, 0), ax.set_aspect('equal'), ax.axis('off')
    cmap, norm = plt.cm.plasma, mcolors.Normalize(vmin=0, vmax=1000)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05))
    cbar.set_label('Fitness (Q*R)')
    w = 1 / 3.0
    sub_offsets = {4: (0, 0), 2: (w, 0), 5: (2*w, 0), 0: (0, w), 1: (2*w, w), 6: (0, 2*w), 3: (w, 2*w), 7: (2*w, 2*w)}
    arrows = {0: '←', 1: '→', 2: '↑', 3: '↓', 4: '↖', 5: '↗', 6: '↙', 7: '↘'}
    if best_agent:
        for r in range(env.rows):
            for c in range(env.cols):
                ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='black', edgecolor='gray', linewidth=0.5))
                pos = (r, c)
                if pos in env.obstacles:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='#444444', hatch='///'))
                    continue
                if pos == env.goal_pos:
                    ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor='gold'))
                    ax.text(c + 0.5, r + 0.5, 'G', ha='center', va='center', fontweight='bold', fontsize=14)
                    continue
                state_str = [str(r), str(c)]
                match_set = [cl for cl in best_agent.population if cl.matches(state_str)]
                action_fitness = {a: 0.0 for a in range(8)}
                for cl in match_set:
                    if cl.fitness > action_fitness.get(cl.A, -1.0):
                        action_fitness[cl.A] = cl.fitness
                best_action = max(action_fitness, key=action_fitness.get) if action_fitness else -1
                for a_idx, (dx, dy) in sub_offsets.items():
                    fit = action_fitness.get(a_idx, 0.0)
                    if fit > 0:
                        ax.add_patch(patches.Rectangle((c + dx, r + dy), w, w, facecolor=cmap(norm(fit)), linewidth=0))
                if best_action != -1 and action_fitness.get(best_action, 0.0) > 50:
                    ax.text(c + 0.5, r + 0.5, arrows.get(best_action, ''), ha='center', va='center', color='black', fontsize=18, fontweight='bold', zorder=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='square,pad=0.02'))

def _plot_top_rules(ax, best_agent, title_prefix: str = ""):
    ax.axis('off'), ax.set_title(f"{title_prefix}Top 10 Rules (Best Agent)")
    table_data = []
    if best_agent:
        top_classifiers = sorted([cl for cl in best_agent.population if cl.fitness > 1.0], key=lambda x: x.fitness, reverse=True)[:10]
        arrows = {0: '←', 1: '→', 2: '↑', 3: '↓', 4: '↖', 5: '↗', 6: '↙', 7: '↘'}
        for cl in top_classifiers:
            mark_str = ", ".join([",".join(list(s)) for s in cl.M]) if cl.M else ""
            table_data.append([cl.C, arrows.get(cl.A, str(cl.A)), cl.E, mark_str, f"{cl.q:.2f}", f"{cl.r:.0f}", f"{int(cl.fitness)}", str(cl.num)])
    if not table_data:
        table_data = [["-"] * 8]
    ax.table(cellText=table_data, colLabels=["C", "A", "E", "M", "Q", "R", "Fit", "Num"], loc='center', bbox=[0, 0, 1, 1])

def _plot_origin_distribution(ax, stats, b1, b2, is_in_dashboard=False, title_prefix: str = ""):
    if 'mean_steps' in stats and len(stats['mean_steps']) > 0:
        num_episodes = len(stats['mean_steps'])
        episodes = np.arange(num_episodes)
    else:
        ax.set_title(f"{title_prefix}Classifier Origin Distribution Over Time (No data)")
        return

    labels = ['Covering (run_step) %', 'GA (ga_evolve) %', 'ALP Unexpected %', 'ALP Expected %', 'ALP Covering %']
    keys = ['mean_covering_perc', 'mean_ga_perc', 'mean_alp_unexpected_perc', 'mean_alp_expected_perc', 'mean_alp_covering_perc']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    y_values_list = []
    for key in keys:
        data = stats.get(key, np.array([]))
        if data.size > 0:
            y_values_list.append(data.flatten()[:num_episodes])
        else:
            y_values_list.append(np.zeros(num_episodes))

    y_values = np.array(y_values_list)
    
    ax.stackplot(episodes, y_values, labels=labels, colors=colors, alpha=0.8)
    ax.set_title(f"{title_prefix}Classifier Origin Distribution Over Time"), ax.set_xlabel("Episode"), ax.set_ylabel("Percentage (%)"), ax.set_ylim(0, 100)
    
    if is_in_dashboard:
        ax.legend(fontsize='x-small', loc='best')
    else:
        ax.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1, 0.5))

    ax.grid(True, alpha=0.3), ax.axvline(x=b1, color='orange', linestyle='--'), ax.axvline(x=b2, color='orange', linestyle='--')


def _plot_origin_distribution_abs(ax, stats, b1, b2, is_in_dashboard=False, title_prefix: str = ""):
    if 'mean_steps' in stats and len(stats['mean_steps']) > 0:
        num_episodes = len(stats['mean_steps'])
        episodes = np.arange(num_episodes)
    else:
        ax.set_title(f"{title_prefix}Absolute Classifier Origin Distribution Over Time (No data)")
        return

    labels = ['Covering (run_step)', 'GA (ga_evolve)', 'ALP Unexpected', 'ALP Expected', 'ALP Covering']
    keys = ['mean_covering_abs', 'mean_ga_abs', 'mean_alp_unexpected_abs', 'mean_alp_expected_abs', 'mean_alp_covering_abs']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    y_values_list = []
    for key in keys:
        data = stats.get(key, np.array([]))
        if data.size > 0:
            y_values_list.append(data.flatten()[:num_episodes])
        else:
            y_values_list.append(np.zeros(num_episodes))

    y_values = np.array(y_values_list)
    
    for i, (line_data, label, color) in enumerate(zip(y_values, labels, colors)):
        ax.plot(episodes, line_data, label=label, color=color)

    ax.set_title(f"{title_prefix}Absolute Classifier Origin Distribution Over Time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of Classifiers")
    
    if is_in_dashboard:
        ax.legend(fontsize='x-small', loc='best')
    else:
        ax.legend(fontsize='x-small', loc='center left', bbox_to_anchor=(1, 0.5))

    ax.grid(True, alpha=0.3)
    ax.axvline(x=b1, color='orange', linestyle='--')
    ax.axvline(x=b2, color='orange', linestyle='--')

def plot_grouped_bar_chart(ax, stats: Dict[str, Any], origin_key: str, title: str, params_phases: Dict[str, Any], plot_absolute_annotations: bool = False, title_prefix: str = ""):
    mean_dist_data = stats.get(f'mean_{origin_key}_creation_dist', np.zeros((10, 10)))
    
    if plot_absolute_annotations:
        abs_dist_data = stats.get(f'mean_{origin_key}_creation_dist_abs', np.zeros((10, 10)))

    TOTAL_EP = sum(p['episodes'] for p in params_phases.values())
    if TOTAL_EP == 0: TOTAL_EP = 1
    
    interval_size = TOTAL_EP / 10
    observation_labels = [f"Ep {int((i+1)*interval_size)}" for i in range(10)]
    creation_labels = [f"Ep {int(i*interval_size)+1}-{int((i+1)*interval_size)}" for i in range(10)]
    
    cmap = plt.cm.get_cmap('YlGnBu', 10)
    creation_colors = [cmap(i) for i in range(10)]
    
    n_groups = 10
    n_bars_per_group = 10
    bar_width = 0.8 / n_bars_per_group
    
    x = np.arange(n_groups)
    
    all_handles, all_labels = [], []
    for i in range(n_bars_per_group):
        bars = mean_dist_data[:, i]
        bar_containers = ax.bar(x + i * bar_width, bars, bar_width, label=creation_labels[i], color=creation_colors[i])
        
        if bar_containers.patches:
            all_handles.append(bar_containers.patches[0])
            all_labels.append(creation_labels[i])
        
        if plot_absolute_annotations:
            abs_bars = abs_dist_data[:, i]
            for j, bar in enumerate(bar_containers):
                height = bar.get_height()
                abs_value = abs_bars[j]
                if abs_value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1, # +1 to place above bar
                            f'{int(abs_value)}', ha='center', va='bottom', fontsize=7, color='black')
        
    ax.set_title(f"{title_prefix}{title}")
    ax.set_xlabel("Observation Episode")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x + (n_bars_per_group / 2 - 0.5) * bar_width)
    ax.set_xticklabels(observation_labels, rotation=45, ha='right', fontsize='small')
    ax.grid(axis='y', alpha=0.3)
    return all_handles, all_labels


def create_loaded_dashboard(loaded_stats: Dict[str, Any],
                            optimal_avg_steps: float,
                            params_phases: Dict[str, Any],
                            n_exp: int,
                            n_steps: int,
                            plot_steps: bool = False, plot_population: bool = False, plot_knowledge: bool = False, plot_reward_quality: bool = False,
                            plot_policy_map: bool = False, plot_top_rules: bool = False, plot_origin_distribution: bool = False,
                            plot_origin_distribution_abs: bool = False,
                            plot_creation_dist_key: Optional[str] = None, plot_all_dashboards: bool = False, timestamp: Optional[str] = None, title_prefix: str = ""):
    
    plot_flags = [plot_steps, plot_population, plot_knowledge, plot_reward_quality,
                  plot_policy_map, plot_top_rules, plot_origin_distribution, plot_origin_distribution_abs,
                  plot_creation_dist_key is not None]
    num_plots = sum(plot_flags)

    rows, cols = 7, 7
    start, goal = (0, 0), (0, 6)
    obstacles = [(1, 2), (1, 4), (1, 5), (2, 1), (3, 3), (3, 4), (4, 1), (4, 3), (4, 6), (5, 1), (5, 4), (6, 5)]
    env = GridEnvironment(rows, cols, start, goal, obstacles)

    if not plot_all_dashboards and num_plots == 1:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        
        best_agent = loaded_stats.get('best_agent', None)
        ema_span, TOTAL_EP = 50, sum(p['episodes'] for p in params_phases.values())
        b1, b2 = params_phases['explore']['episodes'], params_phases['explore']['episodes'] + params_phases['exploit1']['episodes']

        plot_name = ""
        if plot_steps:
            _plot_steps(ax, loaded_stats.get('mean_steps', np.array([])).flatten(), calculate_ema(loaded_stats.get('mean_steps', np.array([])).flatten(), ema_span), optimal_avg_steps, loaded_stats.get('Exploit Avg. Steps', loaded_stats.get('Last Avg. Steps', 0)), b1, b2, ema_span, title_prefix=title_prefix)
            plot_name = "steps"
        elif plot_population:
            _plot_population(ax, loaded_stats.get('mean_micro_pop', np.array([])).flatten(), loaded_stats.get('mean_rel_micro_pop', np.array([])).flatten(), loaded_stats.get('mean_macro_pop', np.array([])).flatten(), loaded_stats.get('mean_rel_macro_pop', np.array([])).flatten(), b1, b2, title_prefix=title_prefix)
            plot_name = "population"
        elif plot_knowledge:
            _plot_knowledge(ax, loaded_stats.get('mean_know', np.array([])).flatten(), loaded_stats.get('std_know', np.array([])).flatten(), loaded_stats.get('mean_generalization', np.array([])).flatten(), b1, b2, title_prefix=title_prefix)
            plot_name = "knowledge"
        elif plot_reward_quality:
            _plot_reward_quality(ax, loaded_stats.get('mean_avg_r', np.array([])).flatten(), loaded_stats.get('mean_avg_rel_r', np.array([])).flatten(), loaded_stats.get('mean_avg_q_all', np.array([])).flatten(), loaded_stats.get('mean_avg_q_rel', np.array([])).flatten(), b1, b2, title_prefix=title_prefix)
            plot_name = "reward_quality"
        elif plot_policy_map:
            policy_avg_steps = calculate_policy_avg_len(best_agent, env)
            _plot_policy_map(fig, ax, best_agent, env, policy_avg_steps, title_prefix=title_prefix)
            plot_name = "policy_map"
        elif plot_top_rules:
            _plot_top_rules(ax, best_agent, title_prefix=title_prefix)
            plot_name = "top_rules"
        elif plot_origin_distribution:
            _plot_origin_distribution(ax, loaded_stats, b1, b2, is_in_dashboard=False, title_prefix=title_prefix)
            plot_name = "origin_distribution"
        elif plot_origin_distribution_abs:
            _plot_origin_distribution_abs(ax, loaded_stats, b1, b2, is_in_dashboard=False, title_prefix=title_prefix)
            plot_name = "origin_distribution_abs"
        elif plot_creation_dist_key:
            handles, labels = plot_grouped_bar_chart(ax, loaded_stats, plot_creation_dist_key, f"{plot_creation_dist_key.upper()} Creation Distribution", params_phases, plot_absolute_annotations=True, title_prefix=title_prefix) # Always true for single plot creation dist
            ax.legend(handles, labels, title="Creation Interval", fontsize='x-small', loc='best')
            plot_name = f"creation_dist_{plot_creation_dist_key}"

        plt.tight_layout()
        
        filename = f'./reports/acs2_project_dashboard_{timestamp}_{plot_name}.png'
        plt.savefig(filename)
        print(f"Dashboard saved to {filename}")
        plt.close('all')
        
    else:
        create_dashboard(best_agent=loaded_stats.get('best_agent', None), optimal_avg_steps=optimal_avg_steps, env=env, stats=loaded_stats, params_phases=params_phases, n_exp=n_exp, n_steps=n_steps, summary_stats=loaded_stats, plot_steps=plot_steps, plot_population=plot_population, plot_knowledge=plot_knowledge, plot_reward_quality=plot_reward_quality, plot_policy_map=plot_policy_map, plot_top_rules=plot_top_rules, plot_origin_distribution=plot_origin_distribution, plot_origin_distribution_abs=plot_origin_distribution_abs, plot_creation_dist_key=plot_creation_dist_key, plot_all_dashboards=plot_all_dashboards, timestamp=timestamp, title_prefix=title_prefix)

def create_dashboard(best_agent: Optional[ACS2], optimal_avg_steps: float, env: Optional[GridEnvironment], stats: Dict[str, Any], params_phases: Dict[str, Any], n_exp: int, n_steps: int, summary_stats: Dict[str, Any],
                     plot_steps: bool = True, plot_population: bool = True, plot_knowledge: bool = True, plot_reward_quality: bool = True,
                     plot_policy_map: bool = True, plot_top_rules: bool = True, plot_origin_distribution: bool = True,
                     plot_origin_distribution_abs: bool = False, # New parameter
                     plot_creation_dist_key: Optional[str] = None, plot_all_dashboards: bool = False, timestamp: Optional[str] = None, title_prefix: str = ""):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(f"{title_prefix}ACS2 Dashboard: Final Report", fontsize=16, fontweight='bold')
    
    # Adjust GridSpec for new plots (absolute origin distribution)
    # Original: 9 rows. Now, if plotting origin_distribution_abs, we need more space.
    # Let's make it 10 rows and use 2 for origin distribution plots (percentage and absolute)
    # The last row is for the text report
    gs = GridSpec(10, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8]) # Increased to 10 rows
    ema_span = 50
    TOTAL_EP = sum(p['episodes'] for p in params_phases.values())
    b1, b2 = params_phases['explore']['episodes'], params_phases['explore']['episodes'] + params_phases['exploit1']['episodes']

    if plot_all_dashboards or plot_steps:
        _plot_steps(fig.add_subplot(gs[0, 0]), stats.get('mean_steps', np.array([])).flatten(), calculate_ema(stats.get('mean_steps', np.array([])).flatten(), ema_span), optimal_avg_steps, summary_stats.get('Exploit Avg. Steps', summary_stats.get('Last Avg. Steps', 0)), b1, b2, ema_span, title_prefix=title_prefix)
    if plot_all_dashboards or plot_population:
        _plot_population(fig.add_subplot(gs[0, 1]), stats.get('mean_micro_pop', np.array([])).flatten(), stats.get('mean_rel_micro_pop', np.array([])).flatten(), stats.get('mean_macro_pop', np.array([])).flatten(), stats.get('mean_rel_macro_pop', np.array([])).flatten(), b1, b2, title_prefix=title_prefix)
    if plot_all_dashboards or plot_knowledge:
        _plot_knowledge(fig.add_subplot(gs[0, 2]), stats.get('mean_know', np.array([])).flatten(), stats.get('std_know', np.array([])).flatten(), stats.get('mean_generalization', np.array([])).flatten(), b1, b2, title_prefix=title_prefix)
    if plot_all_dashboards or plot_reward_quality:
        _plot_reward_quality(fig.add_subplot(gs[0, 3]), stats.get('mean_avg_r', np.array([])).flatten(), stats.get('mean_avg_rel_r', np.array([])).flatten(), stats.get('mean_avg_q_all', np.array([])).flatten(), stats.get('mean_avg_q_rel', np.array([])).flatten(), b1, b2, title_prefix=title_prefix)
    if (plot_all_dashboards or plot_policy_map) and best_agent and env:
        policy_avg_steps = calculate_policy_avg_len(best_agent, env)
        _plot_policy_map(fig, fig.add_subplot(gs[1, 0:2]), best_agent, env, policy_avg_steps, title_prefix=title_prefix)
    if (plot_all_dashboards or plot_top_rules) and best_agent:
        _plot_top_rules(fig.add_subplot(gs[1, 2:4]), best_agent, title_prefix=title_prefix)
    
    # Origin Distribution Plots (Percentage and Absolute)
    # These will now share row 2
    if plot_all_dashboards or plot_origin_distribution:
        _plot_origin_distribution(fig.add_subplot(gs[2, 0:2]), stats, b1, b2, is_in_dashboard=True, title_prefix=title_prefix)
    if plot_all_dashboards or plot_origin_distribution_abs: # New plot
        _plot_origin_distribution_abs(fig.add_subplot(gs[2, 2:4]), stats, b1, b2, is_in_dashboard=True, title_prefix=title_prefix)
    
    creation_plot_keys = ["alp_expected", "covering", "ga", "alp_unexpected", "alp_covering"]
    all_grouped_handles, all_grouped_labels = [], []

    if plot_all_dashboards or plot_creation_dist_key in creation_plot_keys:
        for i, key in enumerate(creation_plot_keys):
            if plot_all_dashboards or key == plot_creation_dist_key:
                # Shifted down by 1 row due to new plot in row 2
                ax = fig.add_subplot(gs[3+i, :]) # gs[3] is now the first creation distribution plot
                handles, labels = plot_grouped_bar_chart(ax, stats, key, f"{key.upper()} Creation Distribution", params_phases, plot_absolute_annotations=plot_all_dashboards, title_prefix=title_prefix) # Pass flag here
                
                if not all_grouped_handles:
                    all_grouped_handles = handles
                    all_grouped_labels = labels

                if not plot_all_dashboards or i < len(creation_plot_keys) - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)

    if plot_all_dashboards and all_grouped_handles:
        legend_ax = fig.add_subplot(gs[8, :])
        legend_ax.legend(all_grouped_handles, all_grouped_labels, title="Creation Interval", ncol=5, loc='center')
        legend_ax.axis('off')

    ax6 = fig.add_subplot(gs[9, :]) # Shifted to 10th row (index 9)
    ax6.axis('off')
    
    if best_agent:
        total_time = summary_stats.get('Total Time', 0.0)
        avg_time = summary_stats.get('Avg Time', 0.0)
        std_time = summary_stats.get('Std Time', 0.0)

        text_report = f"""--- PARAMETER & RESULT SUMMARY ---
{'='*110}
[GENERAL] N_EXP: {n_exp} | MAX_STEPS: {n_steps} | u_max: {best_agent.cfg.u_max} | Theta_GA: {best_agent.cfg.theta_ga}
          Gamma: {best_agent.cfg.gamma} | Beta: {best_agent.cfg.beta} | Theta_i: {best_agent.cfg.theta_i} | Theta_r: {best_agent.cfg.theta_r} | Theta_AS: {best_agent.cfg.theta_as}
[GA]      Status: {'ENABLED (Mu={best_agent.cfg.mu}, Chi={best_agent.cfg.chi}, Theta_Exp={best_agent.cfg.theta_exp})' if any(p.get('ga', False) for p in params_phases.values()) else 'DISABLED'}
[TIME]    Total: {total_time:.2f}s | Avg: {avg_time:.2f}s ± {std_time:.2f}s
{'-'*110}
{'PHASE':<12} | {'EPISODES':<10} | {'EPSILON':<10} | {'BETA':<10} | {'ALP':<6} | {'GA':<6} | {'DECAY':<6}
"""
        phase_order = ['explore', 'exploit1', 'exploit2']
        for name in phase_order:
            p = params_phases.get(name)
            if p:
                text_report += f"{name:<12} | {p['episodes']:<10} | {p['epsilon']:<10} | {p['beta']:<10} | {str(p.get('alp', False)):<6} | {str(p.get('ga', False)):<6} | {str(p.get('decay', False)):<6}\n"
        
        text_report += f"{'-'*110}\n[RESULTS - AVERAGES FROM EXPLOIT2]\n"
        policy_avg_steps = calculate_policy_avg_len(best_agent, env)
        text_report += f" >> Steps (Exploit Avg.): {summary_stats.get('Exploit Avg. Steps', summary_stats.get('Last Avg. Steps', 0)):.2f} vs Optimal: {optimal_avg_steps:.2f} vs Best Policy: {policy_avg_steps:.2f}\n"
        text_report += f" >> Knowledge: {summary_stats.get('Knowledge', 0) * 100:.2f}% | Generalization: {summary_stats.get('Generalization', 0) * 100:.2f}%\n"
        text_report += f" >> Avg Reward (All): {summary_stats.get('Rew (All)', 0):.2f} | Avg Reward (Rel): {summary_stats.get('Rew (Rel)', 0):.2f}\n"
        text_report += f" >> Population Micro: {summary_stats.get('Micro', 0):.1f} (Rel: {summary_stats.get('Micro (Rel)', 0):.1f}) | Macro: {summary_stats.get('Macro', 0):.1f} (Rel: {summary_stats.get('Macro (Rel)', 0):.1f})\n"
        ax6.text(0.01, 1.0, text_report, transform=ax6.transAxes, fontsize=9, fontfamily='monospace', va='top')


    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    plot_names = []
    if plot_steps: plot_names.append("steps")
    if plot_population: plot_names.append("population")
    if plot_knowledge: plot_names.append("knowledge")
    if plot_reward_quality: plot_names.append("reward_quality")
    if plot_policy_map: plot_names.append("policy_map")
    if plot_top_rules: plot_names.append("top_rules")
    if plot_origin_distribution: plot_names.append("origin_distribution")
    if plot_creation_dist_key: plot_names.append(f"creation_dist_{plot_creation_dist_key}")
    if plot_all_dashboards: plot_names = ["all"]

    plot_name_str = "_".join(plot_names) if plot_names else "unspecified"

    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    filename = f'./reports/acs2_project_dashboard_{timestamp}_{plot_name_str}.png'
    plt.savefig(filename)
    print(f"Dashboard saved to {filename}")
    plt.close('all')
