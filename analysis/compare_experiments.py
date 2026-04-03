import argparse
import os
import pandas as pd
import glob
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from plotting import plot_comparison

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results.")
    parser.add_argument('result_dirs', nargs='+', type=str, help='List of experiment result directories to compare.')
    parser.add_argument('--metric', type=str, default='steps_to_goal', help='Metric to compare.')
    args = parser.parse_args()

    all_data = []

    for result_dir in args.result_dirs:
        if not os.path.exists(result_dir):
            print(f"Warning: Directory not found: {result_dir}")
            continue

        csv_files = glob.glob(os.path.join(result_dir, '*.csv'))
        
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                basename = os.path.basename(f)
                model_name = basename.split('_run_')[0]
                run_num = int(basename.split('_run_')[1].replace('.csv', ''))

                df['model_name'] = model_name
                df['run_num'] = run_num
                all_data.append(df)
            except Exception as e:
                print(f"Error reading file {f}: {e}")

    if not all_data:
        print("No data to compare.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Determine output directory
    if len(args.result_dirs) == 1:
        output_dir = args.result_dirs[0]
    else:
        output_dir = '.' # Default to current directory if multiple dirs are compared

    output_filename = os.path.join(output_dir, f"comparison_{args.metric}.png")
    plot_comparison(full_df, args.metric, output_filename)

    # Statistical tests
    model_names = full_df['model_name'].unique()
    if len(model_names) == 2:
        print(f"\n--- T-test for metric: {args.metric} (last episode) ---")
        
        # Get data for the last episode for each model
        last_episode_df = full_df[full_df['episode'] == full_df['episode'].max()]
        
        data1 = last_episode_df[last_episode_df['model_name'] == model_names[0]][args.metric]
        data2 = last_episode_df[last_episode_df['model_name'] == model_names[1]][args.metric]

        if len(data1) > 1 and len(data2) > 1:
            t_stat, p_value = stats.ttest_ind(data1, data2)
            print(f"Model 1: {model_names[0]}, Mean: {data1.mean():.2f}")
            print(f"Model 2: {model_names[1]}, Mean: {data2.mean():.2f}")
            print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("The difference is statistically significant (p < 0.05).")
            else:
                print("The difference is not statistically significant (p >= 0.05).")
        else:
            print("Not enough data for t-test.")
    else:
        print("\nStatistical test (t-test) is only performed for two models.")


if __name__ == '__main__':
    main()