import os
import json
import pandas as pd
from scipy.stats import ttest_ind


def generate_latex_table(df, model_order):
    """Generates a LaTeX table from a DataFrame with mean/std results and significance.

    :param df: DataFrame containing model performance data.
    :type df: pandas.DataFrame
    :param model_order: A list of model names in the desired display order.
    :type model_order: list
    :returns: A string containing the complete LaTeX table.
    :rtype: str
    """
    header = r"""\begin{table*}[t]
\centering
\caption{Comparing Model Performance on Benchmark Datasets. Values are mean $\pm$ std over 30 runs. A (*) indicates a result is statistically significantly better than the Decision Tree baseline (p < 0.05).}
\label{tab:performance_comparison}
\begin{tabular}{l l c c c c c}
\toprule
\textbf{Dataset} & \textbf{Model} & \textbf{MAE} & \textbf{QWK} & \textbf{MSE} & \textbf{Accuracy} & \textbf{Balanced Acc.} \\
\midrule"""
    footer = r"""\bottomrule
\end{tabular}
\end{table*}"""
    
    body = []
    
    dataset_order = ['Car Evaluation', 'Wine Quality', 'Boston Housing', 'Poker Hand']
    df['dataset_order'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)
    df['model_order'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values(by=['dataset_order', 'model_order'])

    current_dataset = None
    for _, row in df.iterrows():
        if row['Dataset'] != current_dataset:
            if current_dataset is not None:
                body.append(r'\midrule')
            current_dataset = row['Dataset']
            body.append(r'\multicolumn{7}{l}{\textit{' + current_dataset + r'}} \\')

        metrics_to_format = ['MAE', 'QWK', 'MSE', 'Accuracy', 'Balanced Acc.']
        formatted_metrics = []
        for metric in metrics_to_format:
            mean_val = row.get(f'{metric}_mean')
            std_val = row.get(f'{metric}_std')
            sig_val = row.get(f'{metric}_sig', '')
            if pd.notna(mean_val) and pd.notna(std_val):
                formatted_metrics.append(f"{mean_val:.4f} $\pm$ {std_val:.4f}{sig_val}")
            else:
                formatted_metrics.append('---')

        row_values = [row['Model']] + formatted_metrics
        body.append(' & '.join(row_values) + r' \\')

    return '\n'.join([header, '\n'.join(body), footer])


if __name__ == '__main__':
    # --- Data Loading and Structuring ---
    model_display_map = {
        'DecisionTree': 'Decision Tree',
        'SVM': 'SVM',
        'CLM': 'CLM (Ordinal Ridge)',
        'MLP-POM': 'POM (MLP Base)',
        'MLP-Adjacent': 'Adjacent (MLP Base)',
        'MLP-MLP': 'MLP (Classification)',
        'MLP-MLP-EMD': 'MLP (EMD Loss)',
        'MLP-CORAL': 'CORAL (MLP Base)',
        'MLP-CORN': 'CORN (MLP Base)',
    }
    model_order = list(model_display_map.values())
    datasets = {
        'Car': 'Car Evaluation',
        'Wine': 'Wine Quality',
        'Boston': 'Boston Housing',
        'Poker': 'Poker Hand',
    }

    table_rows = []
    for script_ds, display_ds in datasets.items():
        for script_model, display_model in model_display_map.items():
            table_rows.append({
                'Dataset': display_ds,
                'Model': display_model,
                'script_name': script_model
            })
    main_df = pd.DataFrame(table_rows)

    results_dir = 'results'
    if not os.path.exists(results_dir): exit()
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    raw_results_dict = {}
    for file_name in result_files:
        with open(os.path.join(results_dir, file_name), 'r') as f:
            data = json.load(f)
            run_model_name = data.get('model_name')
            run_dataset_name = data.get('dataset')
            agg_metrics = data.get('aggregated_metrics', {})
            raw_results = data.get('raw_results', [])

            match_condition = (main_df['script_name'] == run_model_name) & (main_df['Dataset'] == datasets.get(run_dataset_name))
            if match_condition.any():
                idx_to_update = main_df.index[match_condition].tolist()[0]
                raw_results_dict[(run_dataset_name, run_model_name)] = raw_results
                for metric, values in agg_metrics.items():
                    main_df.loc[idx_to_update, f'{metric}_mean'] = values.get('mean')
                    main_df.loc[idx_to_update, f'{metric}_std'] = values.get('std')

    # --- Statistical Significance Testing ---
    baseline_model_script_name = 'DecisionTree'
    metrics_to_test = ['MAE', 'QWK', 'MSE', 'Accuracy', 'Balanced Acc.']
    higher_is_better = ['QWK', 'Accuracy', 'Balanced Acc.']

    for dataset_script_name in datasets.keys():
        baseline_key = (dataset_script_name, baseline_model_script_name)
        if baseline_key not in raw_results_dict:
            continue
        baseline_results_df = pd.DataFrame(raw_results_dict[baseline_key])

        for model_script_name in model_display_map.keys():
            if model_script_name == baseline_model_script_name:
                continue
            
            model_key = (dataset_script_name, model_script_name)
            if model_key not in raw_results_dict:
                continue
            model_results_df = pd.DataFrame(raw_results_dict[model_key])

            for metric in metrics_to_test:
                if metric not in baseline_results_df.columns or metric not in model_results_df.columns:
                    continue

                baseline_scores = baseline_results_df[metric].dropna()
                model_scores = model_results_df[metric].dropna()

                if len(baseline_scores) < 2 or len(model_scores) < 2:
                    continue

                _, p_value = ttest_ind(model_scores, baseline_scores, equal_var=False)

                if p_value < 0.05:
                    is_better = (model_scores.mean() > baseline_scores.mean()) if metric in higher_is_better else (model_scores.mean() < baseline_scores.mean())
                    if is_better:
                        df_match_condition = (main_df['script_name'] == model_script_name) & (main_df['Dataset'] == datasets[dataset_script_name])
                        idx_to_update = main_df.index[df_match_condition].tolist()[0]
                        main_df.loc[idx_to_update, f'{metric}_sig'] = '*'

    final_df = main_df.dropna(subset=['MAE_mean', 'MSE_mean'], how='all').copy()
    
    latex_output = generate_latex_table(final_df, model_order)
    print(latex_output)
