import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results_to_dataframe():
    """Loads all JSON results from the results directory into a pandas DataFrame."""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return pd.DataFrame()

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    all_results = []
    for file_name in result_files:
        with open(os.path.join(results_dir, file_name), 'r') as f:
            data = json.load(f)
            # Skip regression results for this chart
            if data['dataset'] == 'Boston' and 'QWK' not in data['aggregated_metrics']:
                continue
            
            result_row = {
                'model': data.get('model_name'),
                'dataset': data.get('dataset'),
                'timestamp': data.get('timestamp', '19700101_000000'), # Default for old files
                **{f'{metric}': values['mean'] for metric, values in data.get('aggregated_metrics', {}).items()}
            }
            all_results.append(result_row)
            
    df = pd.DataFrame(all_results)
    
    if df.empty:
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
        'DecisionTree': 'Decision Tree',
        'SVM': 'SVM',
    }
    df['model'] = df['model'].replace(name_map)

    # Sort by timestamp and drop duplicates, keeping only the most recent result
    df = df.sort_values(by='timestamp', ascending=False)
    df = df.drop_duplicates(subset=['model', 'dataset'], keep='first')
        
    return df

def create_grouped_bar_chart(df, metric, output_filename):
    """Creates and saves a grouped bar chart for a given metric."""
    if df.empty or metric not in df.columns:
        print(f"No data available to plot for metric: {metric}")
        return

    # We only want to plot the MLP-based models for a clean comparison
    # For this chart, we'll include all models that have results
    plot_df = df.copy()

    # Pivot the data for plotting
    pivot_df = plot_df.pivot(index='dataset', columns='model', values=metric)
    pivot_df = pivot_df.dropna(how='all') # Drop datasets with no data

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
    plt.savefig(output_filename)
    print(f"Chart saved to {output_filename}")
    plt.close() # Close the plot to free up memory

if __name__ == '__main__':
    results_df = load_results_to_dataframe()
    metrics_to_plot = ['QWK', 'MAE', 'MSE', 'Accuracy', 'Balanced Acc.']
    
    for metric in metrics_to_plot:
        output_file = os.path.join('figures', f'{metric.lower().replace(" ", "_")}_summary_chart.png')
        print(f"\nGenerating chart for {metric}...")
        create_grouped_bar_chart(results_df.copy(), metric=metric, output_filename=output_file)