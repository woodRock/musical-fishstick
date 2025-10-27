"""Creates a bar chart comparing the performance of different models on a given dataset.

This script reads a set of JSON files from the 'results' directory, each representing
the performance of a model. It then extracts a specified metric (e.g., 'MAE', 'MSE')
and generates a bar chart to visualize and compare the models' performance.
The resulting chart is saved in the 'figures' directory.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results_to_dataframe():
    """Loads all JSON results from the results directory into a pandas DataFrame.

    This function scans the 'results' directory for JSON files, parses them,
    and consolidates the evaluation metrics into a single pandas DataFrame.
    It filters out regression results for the 'Boston' dataset if 'QWK' is not present,
    and cleans up model names for consistent display. Only the most recent result
    for each model-dataset pair is kept.

    Returns:
        pd.DataFrame: A DataFrame containing the consolidated results,
                      or an empty DataFrame if no results are found or processed.
    """
    results_dir = '/Users/woodj/Desktop/musical-fishstick/results'
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return pd.DataFrame()

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    all_results = []
    for file_name in result_files:
        with open(os.path.join(results_dir, file_name), 'r') as f:
            data = json.load(f)
            # Skip regression results for this chart if QWK is not present
            eval_metrics = data.get('aggregated_metrics', {})
            if data['dataset'].title() == 'Boston' and 'QWK' not in eval_metrics:
                continue
            
            result_row = {
                'model': data.get('model_name'),
                'dataset': data.get('dataset').title(), # Normalize dataset name to title case
                'timestamp': data.get('timestamp', '19700101_000000'), # Default for old files
            }
            for metric_name, metric_values in eval_metrics.items():
                if 'mean' in metric_values:
                    result_row[metric_name] = metric_values['mean']
                else:
                    # Handle cases where 'mean' might not be present, e.g., for raw values
                    result_row[metric_name] = metric_values
            all_results.append(result_row)
            
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("DataFrame is empty after loading all results.")
        return df

    # Clean up model names for display before dropping duplicates
    name_map = {
        'MLP-POM': 'POM (MLP)',
        'Linear-POM': 'POM (Linear)',
        'MLP-Adjacent': 'Adjacent (MLP)',
        'Linear-Adjacent': 'Adjacent (Linear)',
        'MLP-MLP': 'MLP',
        'MLP-MLP-EMD': 'MLP (EMD)',
        'MLP-CORAL': 'CORAL',
        'MLP-CORN': 'CORN',
        'CLM': 'CLM',
        'SVM': 'SVM',
        'SVOR': 'SVOR'
    }
    df['model'] = df['model'].replace(name_map)

    # Sort by timestamp and drop duplicates, keeping only the most recent result
    df = df.sort_values(by='timestamp', ascending=False)
    df = df.drop_duplicates(subset=['model', 'dataset'], keep='first')
        
    return df

def create_grouped_bar_chart(df, metric, output_filename):
    """Creates and saves a grouped bar chart for a given metric.

    This function filters the input DataFrame to include only MLP-based models,
    pivots the data, and then generates a grouped bar chart comparing model
    performance across different datasets for the specified metric. The chart
    is saved as a PNG image.

    Args:
        df (pd.DataFrame): The input DataFrame containing model results.
        metric (str): The name of the metric to plot (e.g., 'QWK', 'MAE').
        output_filename (str): The path and filename to save the generated chart.
    """
    if df.empty or metric not in df.columns:
        print(f"No data available to plot for metric: {metric}")
        return

    # Include SVOR and SVM for comparison with MLP-based models
    models_to_plot = ['POM (MLP)', 'Adjacent (MLP)', 'MLP', 'MLP (EMD)', 'CORAL', 'CORN', 'SVM', 'SVOR']
    plot_df = df[df['model'].isin(models_to_plot)]

    print(f"Plot DataFrame for {metric} before pivoting:\n{plot_df.to_string()}")

    # Pivot the data for plotting
    pivot_df = plot_df.pivot(index='dataset', columns='model', values=metric)
    pivot_df = pivot_df.dropna(how='all') # Drop datasets with no data

    print(f"Pivot DataFrame for {metric} before plotting:\n{pivot_df.to_string()}")

    if pivot_df.empty:
        print(f"No data to plot for metric '{metric}' after filtering.")
        return

    # Plotting
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8, colormap='viridis')

    # Formatting
    plt.title(f'Comparison of Model Performance by {metric}', fontsize=16)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Dataset', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    # Save the figure
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_filename)
    print(f"Chart saved to {output_filename}")
    plt.close() # Close the plot to free up memory

if __name__ == '__main__':
    results_df = load_results_to_dataframe()
    metrics_to_plot = ['QWK', 'MAE', 'MSE', 'Accuracy', 'Balanced Acc.']
    
    for metric in metrics_to_plot:
        output_file = os.path.join('../../figures', f'{metric.lower().replace(" ", "_")}_summary_chart.png')
        print(f"\nGenerating chart for {metric}...")
        create_grouped_bar_chart(results_df.copy(), metric=metric, output_filename=output_file)
