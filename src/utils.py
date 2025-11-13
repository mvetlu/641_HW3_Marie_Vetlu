import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


RANDOM_SEED = 42
def set_seeds(seed=RANDOM_SEED):
    # random seeds for reproducibility
    np.random.seed(seed) # for numpy
    random.seed(seed) # random
    tf.random.set_seed(seed) # and tensorflow


def plot_training_history(history, exp_name, save_dir='../results/plots'):
    # plot training and validation accuracy/loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # accuracy plot
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{exp_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # loss plot
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title(f'{exp_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{exp_name}_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(results_df, metric='test_accuracy', group_by='architecture', save_dir='../results/plots'):
    # comparison of a metric across different configurations
    plt.figure(figsize=(10, 6))
    grouped = results_df.groupby(group_by)[metric].mean().sort_values(ascending=False)

    sns.barplot(x=grouped.index, y=grouped.values, palette='viridis')
    plt.title(f'{metric.replace("_", " ").title()} by {group_by.title()}')
    plt.xlabel(group_by.title())
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{metric}_by_{group_by}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_comparisons(results_df, save_dir='../results/plots'):
    # generate all comparison plots for the experiment
    factors = ['architecture', 'activation', 'optimizer', 'seq_length', 'gradient_clipping']
    metrics = ['test_accuracy', 'test_f1_score', 'time_per_epoch']

    for factor in factors:
        for metric in metrics:
            plot_metric_comparison(results_df, metric=metric, group_by=factor, save_dir=save_dir)

    print(f"All comparison plots saved to {save_dir}/")


def plot_top_models(results_df, top_n=5, save_dir='../results/plots'):
    # plot the top N models by test accuracy
    plt.figure(figsize=(12, 6))

    top_models = results_df.nsmallest(top_n, 'test_accuracy') if len(results_df) > top_n else results_df
    top_models = top_models.sort_values('test_accuracy', ascending=True)

    y_pos = np.arange(len(top_models))

    plt.barh(y_pos, top_models['test_accuracy'], color='skyblue', alpha=0.8)
    plt.yticks(y_pos, top_models['experiment'])
    plt.xlabel('Test Accuracy')
    plt.title(f'Top {min(top_n, len(results_df))} Models by Test Accuracy')
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/top_models.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_seq_length(results_df, save_dir='../results/plots'):
    # accuracy vs seq len
    seq_experiments = results_df[results_df['experiment'].str.contains('seqlen_')]

    plt.figure(figsize=(10, 6))
    plt.plot(seq_experiments['seq_length'], seq_experiments['test_accuracy'],
             marker='o', linewidth=2, markersize=10, color='blue')
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy vs Sequence Length', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks([25, 50, 100])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_vs_seq_length.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_f1_vs_seq_length(results_df, save_dir='../results/plots'):
    #  F1 vs sequence length
    seq_experiments = results_df[results_df['experiment'].str.contains('seqlen_')]

    plt.figure(figsize=(10, 6))
    plt.plot(seq_experiments['seq_length'], seq_experiments['test_f1_score'],
             marker='o', linewidth=2, markersize=10, color='green')
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('F1-Score (Macro)', fontsize=12)
    plt.title('F1-Score vs Sequence Length', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks([25, 50, 100])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/f1_vs_seq_length.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_best_worst_loss(training_histories, results_df, save_dir='../results/plots'):
    # training loss vs epochs for best and worst models
    best_model = results_df.loc[results_df['test_accuracy'].idxmax(), 'experiment']
    worst_model = results_df.loc[results_df['test_accuracy'].idxmin(), 'experiment']

    best_history = training_histories[best_model]
    worst_history = training_histories[worst_model]

    plt.figure(figsize=(12, 6))

    epochs_best = range(1, len(best_history['loss']) + 1)
    epochs_worst = range(1, len(worst_history['loss']) + 1)

    plt.plot(epochs_best, best_history['loss'], label=f'Best Model ({best_model}) - Train',
             linewidth=2, color='blue', linestyle='-')
    plt.plot(epochs_best, best_history['val_loss'], label=f'Best Model ({best_model}) - Val',
             linewidth=2, color='blue', linestyle='--')

    plt.plot(epochs_worst, worst_history['loss'], label=f'Worst Model ({worst_model}) - Train',
             linewidth=2, color='red', linestyle='-')
    plt.plot(epochs_worst, worst_history['val_loss'], label=f'Worst Model ({worst_model}) - Val',
             linewidth=2, color='red', linestyle='--')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss: Best vs Worst Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/best_worst_loss.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_summary_stats(results_df, save_path='../results/summary_stats.txt'):
    # summary statistics
    with open(save_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EXPERIMENT SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Overall Statistics:\n")
        f.write(f"Total Experiments: {len(results_df)}\n")
        f.write(f"Mean Test Accuracy: {results_df['test_accuracy'].mean():.4f}\n")
        f.write(f"Std Test Accuracy: {results_df['test_accuracy'].std():.4f}\n")
        f.write(f"Mean F1-Score: {results_df['test_f1_score'].mean():.4f}\n")
        f.write(f"Mean Time per Epoch: {results_df['time_per_epoch'].mean():.2f}s\n\n")

        f.write("Best Model:\n")
        best = results_df.loc[results_df['test_accuracy'].idxmax()]
        f.write(f"Experiment: {best['experiment']}\n")
        f.write(f"Architecture: {best['architecture']}\n")
        f.write(f"Activation: {best['activation']}\n")
        f.write(f"Optimizer: {best['optimizer']}\n")
        f.write(f"Sequence Length: {best['seq_length']}\n")
        f.write(f"Gradient Clipping: {best['gradient_clipping']}\n")
        f.write(f"Test Accuracy: {best['test_accuracy']:.4f}\n")
        f.write(f"Test F1-Score: {best['test_f1_score']:.4f}\n")
        f.write(f"Time per Epoch: {best['time_per_epoch']:.2f}s\n")

    print(f"Summary statistics saved to {save_path}")