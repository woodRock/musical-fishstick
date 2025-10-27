"""Module for evaluating machine learning models on various datasets."""

import argparse
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.models import model_factory
from src import util
from src import dl_trainer

def main():
    """Main function to parse arguments, evaluate models, and save results.

    This function sets up argument parsing for dataset, model types (scikit-learn or PyTorch),
    loads the specified dataset, and performs 30 runs of training and evaluation.
    It calculates various metrics, aggregates them, and saves the results to a JSON file.
    """
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset for 30 runs.")
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset to use.')
    parser.add_argument('--model', type=str, help='The scikit-learn model to evaluate (options: DecisionTree, SVM, CLM).')
    parser.add_argument('--base_model', type=str, help='The base model for PyTorch evaluations (options: Linear, MLP).')
    parser.add_argument('--head_model', type=str, help='The head model for PyTorch evaluations (options: ClassificationHead, OrdinalHead, POM, Adjacent, CORAL, CORN, MLP, MLP-EMD).')
    parser.add_argument('--num_runs', type=int, default=30, help='Number of runs to perform for evaluation (default: 30).')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split (default: 0.2).')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs for deep learning models (default: 20).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for deep learning models (default: 32).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for deep learning models (default: 0.001).')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    print(f"Loading dataset: {args.dataset}...")
    X, y, is_regression = util.load_dataset(args.dataset)

    run_results = []

    for i in range(args.num_runs):
        print(f"--- Starting Run {i+1}/{args.num_runs} ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=i) # Use loop index as random_state

        model_params = {}
        
        if args.base_model and args.head_model:
            model_name = f"{args.base_model}-{args.head_model}"
            model_kwargs = {'input_size': X.shape[1], 'num_classes': int(y.max() + 1)}
            model_params = {"base": args.base_model, "head": args.head_model, **model_kwargs}
            model = model_factory.get_dl_model(args.base_model, args.head_model, is_regression, **model_kwargs)

            if args.head_model.lower() == 'pom':
                model = dl_trainer.train_pom_scratch_model(model, X_train, y_train, model_kwargs['num_classes'], n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.learning_rate)
                y_pred = dl_trainer.predict_pom_scratch(model, X_test)
            elif args.head_model.lower() == 'adjacent':
                model = dl_trainer.train_adjacent_model(model, X_train, y_train, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.learning_rate)
                y_pred = dl_trainer.predict_adjacent(model, X_test)
            elif args.head_model.lower() in ['coral', 'corn']:
                train_func = dl_trainer.train_coral_model if args.head_model.lower() == 'coral' else dl_trainer.train_corn_model
                model = train_func(model, X_train, y_train, model_kwargs['num_classes'], n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.learning_rate)
                y_pred = dl_trainer.predict_coral(model, X_test)
            elif args.head_model.lower() in ['mlp', 'mlp-emd']:
                train_func = dl_trainer.train_emd_model if args.head_model.lower() == 'mlp-emd' else dl_trainer.train_model
                model = train_func(model, X_train, y_train)
                y_pred = dl_trainer.predict(model, X_test)
            # ... (add other DL model training logic here)
            else:
                 raise ValueError(f"Unknown head model: {args.head_model}")

        elif args.model:
            model_name = args.model
            model = model_factory.get_sklearn_model(model_name, is_regression)
            model_params = util.get_serializable_params(model)
            if model_name.lower() in ['pom', 'clm']:
                model.fit(X_train, y_train.to_numpy())
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            raise ValueError("You must specify either --model or both --base_model and --head_model")

        metrics = util.calculate_metrics(y_test, y_pred, is_regression)
        run_results.append(metrics)
        print(f"Run {i+1} Metrics: {metrics}")

    # --- Aggregation and Saving ---
    print("\n--- Aggregating results from 30 runs ---")
    results_df = pd.DataFrame(run_results)
    mean_metrics = results_df.mean().to_dict()
    std_metrics = results_df.std().to_dict()

    aggregated_metrics = {key: {'mean': mean_metrics[key], 'std': std_metrics[key]} for key in mean_metrics}

    filename = f"{model_name}_{args.dataset}.json"
    filepath = os.path.join('results', filename)

    result_data = {
        'model_name': model_name,
        'dataset': args.dataset,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_params': model_params,
        'aggregated_metrics': aggregated_metrics,
        'raw_results': results_df.to_dict(orient='records')
    }

    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4)

    print(f"\nAggregated results saved to {filepath}")
    print("\n--- Final Aggregated Metrics ---")
    for metric, values in aggregated_metrics.items():
        print(f"{metric}: {values['mean']:.4f} \u00b1 {values['std']:.4f}")

if __name__ == "__main__":
    main()
