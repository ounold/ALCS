import os
import numpy as np
import pandas as pd
import yaml # Added for metadata handling
import pickle
from datetime import datetime
from typing import Dict, Any

def export_dashboard_data(stats: Dict[str, Any], summary_stats: Dict[str, float], timestamp: str,
                          params_phases: Dict[str, Any], n_exp: int, n_steps: int, optimal_avg_steps: float, title_prefix: str = ""):
    """
    Exports dashboard data to CSV files and metadata to a YAML file.
    """
    output_dir = os.path.join("reports", "saved_states")
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = title_prefix.strip().replace(" ", "_") + "_" if title_prefix else ""

    # Save stats arrays to CSV
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            filename = f"dashboard_data_{prefix}{timestamp}_{key}.csv"
            filepath = os.path.join(output_dir, filename)
            
            if value.ndim == 3:
                # Reshape 3D array to 2D
                value = value.reshape(value.shape[0], -1)
            elif value.ndim == 1:
                value = value.reshape(-1, 1)

            df = pd.DataFrame(value)
            df.to_csv(filepath, index=False)
            print(f"Saved {key} to {filepath}")
        elif key == 'best_agent':
            filename = f"dashboard_data_{prefix}{timestamp}_best_agent.pkl"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)
            print(f"Saved {key} to {filepath}")

    # Save metadata to YAML
    metadata = {
        'timestamp': timestamp,
        'title_prefix': title_prefix,
        'params_phases': params_phases,
        'n_exp': n_exp,
        'n_steps': n_steps,
        'optimal_avg_steps': optimal_avg_steps,
        'summary_stats': summary_stats, # Add summary_stats here
        'total_episodes': sum(p['episodes'] for p in params_phases.values())
    }
    metadata_filename = f"dashboard_metadata_{prefix}{timestamp}.yaml"
    metadata_filepath = os.path.join(output_dir, metadata_filename)
    with open(metadata_filepath, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"Saved metadata to {metadata_filepath}")

def import_dashboard_data(timestamp: str, title_prefix: str = "") -> (Dict[str, Any], Dict[str, Any]):
    """
    Imports dashboard data from CSV files.
    """
    input_dir = os.path.join("reports", "saved_states")
    
    loaded_stats = {}
    metadata = {}
    
    prefix = title_prefix.strip().replace(" ", "_") + "_" if title_prefix else ""

    # Load metadata
    # Try with prefix first, then without
    metadata_filename = f"dashboard_metadata_{prefix}{timestamp}.yaml"
    metadata_filepath = os.path.join(input_dir, metadata_filename)
    if not os.path.exists(metadata_filepath):
        metadata_filename = f"dashboard_metadata_{timestamp}.yaml"
        metadata_filepath = os.path.join(input_dir, metadata_filename)

    try:
        with open(metadata_filepath, 'r') as f:
            metadata = yaml.safe_load(f)
        print(f"Loaded metadata from {metadata_filepath}")
    except FileNotFoundError:
        print(f"Error: Metadata file {metadata_filepath} not found.")
        return {}, {}
    except Exception as e:
        print(f"Error loading metadata from {metadata_filepath}: {e}")
        return {}, {}
    
    # Update prefix based on loaded metadata if available
    prefix = metadata.get('title_prefix', '').strip().replace(" ", "_") + "_" if metadata.get('title_prefix') else prefix

    # Extract summary_stats from metadata and add to loaded_stats
    summary_stats = metadata.get('summary_stats', {})
    loaded_stats.update(summary_stats)

    # List all files in the directory
    try:
        files = os.listdir(input_dir)
    except FileNotFoundError:
        print(f"Error: Directory {input_dir} not found.")
        return {}, {}

    for f in files:
        if (f.startswith(f"dashboard_data_{prefix}{timestamp}_") or f.startswith(f"dashboard_data_{timestamp}_")) and f.endswith(".csv"):
            key = f.replace(f"dashboard_data_{prefix}{timestamp}_", "").replace(f"dashboard_data_{timestamp}_", "").replace(".csv", "")
            filepath = os.path.join(input_dir, f)
            
            try:
                df = pd.read_csv(filepath)
                loaded_stats[key] = df.values
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return {}, {}
        elif (f.startswith(f"dashboard_data_{prefix}{timestamp}_") or f.startswith(f"dashboard_data_{timestamp}_")) and f.endswith(".pkl"):
            key = f.replace(f"dashboard_data_{prefix}{timestamp}_", "").replace(f"dashboard_data_{timestamp}_", "").replace(".pkl", "")
            filepath = os.path.join(input_dir, f)
            try:
                with open(filepath, 'rb') as pkl_file:
                    loaded_stats[key] = pickle.load(pkl_file)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return {}, {}
                
    return loaded_stats, metadata
