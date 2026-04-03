from __future__ import annotations

import argparse
from datetime import datetime
from src.configGPU4 import build_arg_parserGPU4, experiment_config_from_argsGPU4, load_yaml_defaultsGPU4
from src.universal_runner import UniversalRunner
from src.visualizationCPU3 import create_dashboardCPU3, create_loaded_dashboardCPU3
from src.visualizationGPU4 import create_dashboardGPU4
from src.data_handlerCPU3 import export_dashboard_dataCPU3, import_dashboard_dataCPU3
from src.experiment_runnerCPU3 import calculate_optimal_metricsCPU3, create_environmentCPU3
from environment.runtime_gpu4 import EnvironmentGPU4

def main() -> None:
    # 1. Bootstrap parsing to find config file
    bootstrap_parser = build_arg_parserGPU4({})
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    defaults = load_yaml_defaultsGPU4(bootstrap_args.config)
    
    # 2. Build full parser with defaults and new unified flags
    parser = build_arg_parserGPU4(defaults)
    
    execution_group = parser.add_argument_group("Execution Modes")
    execution_group.add_argument("--explore_mode", type=str, choices=["cpu_single", "cpu_mp", "gpu"], default="gpu",
                                 help="Execution backend for the explore phase.")
    execution_group.add_argument("--exploit_mode", type=str, choices=["cpu_single", "cpu_mp", "gpu"], default="cpu_single",
                                 help="Execution backend for the exploit phases.")
    
    args = parser.parse_args()

    if args.load_dashboard_data:
        print(f"--- Loading dashboard data for timestamp: {args.load_dashboard_data} ---")
        loaded_stats, loaded_metadata = import_dashboard_dataCPU3(args.load_dashboard_data)
        if not loaded_metadata:
            raise FileNotFoundError(
                f"No saved dashboard data found for timestamp '{args.load_dashboard_data}'."
            )

        create_loaded_dashboardCPU3(
            loaded_stats=loaded_stats,
            optimal_avg_steps=loaded_metadata.get("optimal_avg_steps", 0.0),
            params_phases=loaded_metadata.get("params_phases", {}),
            n_exp=loaded_metadata.get("n_exp", 0),
            n_steps=loaded_metadata.get("n_steps", 0),
            environment_metadata=loaded_metadata.get("environment", {}),
            plot_steps=args.plot_steps,
            plot_population=args.plot_population,
            plot_knowledge=args.plot_knowledge,
            plot_reward_quality=args.plot_reward_quality,
            plot_policy_map=args.plot_policy_map,
            plot_top_rules=args.plot_top_rules,
            plot_origin_distribution=args.plot_origin_distribution,
            plot_origin_distribution_abs=args.plot_origin_distribution_abs,
            plot_creation_dist_key=args.plot_creation_dist,
            plot_all_dashboards=args.plot_all_dashboards,
            timestamp=args.load_dashboard_data,
            title_prefix=loaded_metadata.get("title_prefix", ""),
        )
        return

    # 3. Build Configuration
    # Ensure CLI environment_name overrides the config file
    if args.environment_name:
        defaults["environment_name"] = args.environment_name
    experiment_config = experiment_config_from_argsGPU4(args)
    
    # 4. Execute via Universal Runner
    runner = UniversalRunner(experiment_config, args.explore_mode, args.exploit_mode)
    merged_stats, final_summary, best_agent, last_env, optimal_avg_steps = runner.run()
    
    # 5. Dashboard and Data Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title_prefix = f"({args.explore_mode}->{args.exploit_mode}) "
    
    print("--- Generating Unified Dashboard ---")
    plot_all = args.plot_all_dashboards or not any([
        args.plot_steps,
        args.plot_population,
        args.plot_knowledge,
        args.plot_reward_quality,
        args.plot_policy_map,
        args.plot_top_rules,
        args.plot_origin_distribution,
        args.plot_origin_distribution_abs,
        args.plot_creation_dist is not None,
    ])

    if isinstance(last_env, EnvironmentGPU4):
        create_dashboardGPU4(
            best_agent,
            optimal_avg_steps,
            last_env,
            merged_stats,
            experiment_config.params_phases,
            experiment_config.n_exp,
            experiment_config.n_steps,
            final_summary,
            plot_all_dashboards=plot_all,
            timestamp=timestamp,
            title_prefix=title_prefix,
        )
    else:
        create_dashboardCPU3(
            best_agent, 
            optimal_avg_steps, 
            last_env, 
            merged_stats, 
            experiment_config.params_phases, 
            experiment_config.n_exp, 
            experiment_config.n_steps, 
            final_summary, 
            title_prefix=title_prefix, 
            plot_steps=args.plot_steps,
            plot_population=args.plot_population,
            plot_knowledge=args.plot_knowledge,
            plot_reward_quality=args.plot_reward_quality,
            plot_policy_map=args.plot_policy_map,
            plot_top_rules=args.plot_top_rules,
            plot_origin_distribution=args.plot_origin_distribution,
            plot_origin_distribution_abs=args.plot_origin_distribution_abs,
            plot_creation_dist_key=args.plot_creation_dist,
            plot_all_dashboards=plot_all,
            timestamp=timestamp
        )
    
    if args.save_dashboard_data:
        export_dashboard_dataCPU3(merged_stats, final_summary, timestamp, experiment_config, optimal_avg_steps, title_prefix=title_prefix)

if __name__ == "__main__":
    main()
