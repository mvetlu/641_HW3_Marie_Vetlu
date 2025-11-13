import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import f1_score
import os
import sys
from preprocess import get_data
from models import experiments
import pickle
from utils import plot_training_history, plot_all_comparisons, plot_top_models, save_summary_stats, plot_accuracy_vs_seq_length, plot_f1_vs_seq_length, plot_best_worst_loss

sys.path.append('src')
os.makedirs('../results/plots', exist_ok=True) # create directories if they don't exist
data = get_data() # load data from preprocess.py

try: # getting pre-trained data after train.py
    with open('../results/training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)
    training_histories = training_data['histories']
    training_times = training_data['times']
except FileNotFoundError: # sanity check if no pre-trained models found
    print("Error: training_data.pkl not found. Please run train.py first.")
    sys.exit(1)

results = [] # evaluate all models

for exp in experiments:
    exp_name = exp['exp_name']
    config = {k: v for k, v in exp.items() if k != 'exp_name'}
    seq_len = config['seq_length']

    print(f"Evaluating {exp_name}...")
    model = load_model(f'../results/models/{exp_name}_model.keras') # load trained model
    test_loss, test_acc = model.evaluate( # evaluate on test set
        data[f'X_test_{seq_len}'],
        data['y_test'],
        verbose=0
    )

    # getting F1-score
    predictions = model.predict(data[f'X_test_{seq_len}'], verbose=0)
    predictions_binary = (predictions > 0.5).astype(int).flatten()
    f1 = f1_score(data['y_test'], predictions_binary, average='macro')

    # training metrics
    history = training_histories[exp_name]
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    time_per_epoch = training_times[exp_name]

    results.append({
        'experiment': exp_name,
        'architecture': config['architecture'],
        'activation': config['activation'],
        'optimizer': config['optimizer'],
        'seq_length': config['seq_length'],
        'gradient_clipping': config['gradient_clipping'],
        'test_accuracy': test_acc,
        'test_f1_score': f1,
        'time_per_epoch': time_per_epoch,
        'test_loss': test_loss,
        'train_accuracy': final_train_acc,
        'val_accuracy': final_val_acc,
        'train_loss': final_train_loss,
        'val_loss': final_val_loss
    })

# for the results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_accuracy', ascending=False)
results_df.to_csv('../results/metrics.csv', index=False)

# plotting, but not scheming hehe
print("\nGenerating plots...")
for exp in experiments:
    exp_name = exp['exp_name']
    history = training_histories[exp_name]
    plot_training_history(history, exp_name)

plot_all_comparisons(results_df)
plot_top_models(results_df, top_n=5)


save_summary_stats(results_df) # quick summary save
print("\n=== EXPERIMENT RESULTS ===\n") # and print
print(results_df[['experiment', 'test_accuracy', 'test_f1_score', 'time_per_epoch']].to_string(index=False))
print(f"\n\nBest Model (by accuracy): {results_df.iloc[0]['experiment']}")
print(f"Test Accuracy: {results_df.iloc[0]['test_accuracy']:.4f}")
print(f"Test F1-Score: {results_df.iloc[0]['test_f1_score']:.4f}")
print(f"Time per Epoch: {results_df.iloc[0]['time_per_epoch']:.2f}s")