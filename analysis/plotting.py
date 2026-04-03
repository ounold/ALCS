import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_comparison(df: pd.DataFrame, metric: str, output_filename: str):
    """
    Plots the average of a metric over time for different models.
    """
    plt.figure(figsize=(12, 8))
    
    sns.lineplot(data=df, x='episode', y=metric, hue='model_name', errorbar='sd')
    
    plt.title(f'Comparison of Models for Metric: {metric}')
    plt.xlabel('Episode')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(title='Model')
    
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.close()