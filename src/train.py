import numpy as np
import random
import tensorflow as tf
from keras.callbacks import EarlyStopping
import time
import os
import sys
import pickle
from preprocess import get_data
from models import create_model, experiments
from utils import set_seeds

sys.path.append('src')
set_seeds() # using the same random seeds everywhere in the repo
os.makedirs('../results/models', exist_ok=True) # create directories if they don't exist

data = get_data() # load preprocessed data from preprocess.py

# Training parameters
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# early stopping callback
CALLBACKS = [
    EarlyStopping(
        monitor='val_loss',
        patience=3, # stop after 3 epochs with no improvement in validation loss
        restore_best_weights=True
    )
]

# train all 68 experiments
trained_models = {}
training_histories = {}
training_times = {}

if __name__ == '__main__':
    for exp in experiments:
        exp_name = exp['exp_name']
        config = {k: v for k, v in exp.items() if k != 'exp_name'}
        seq_len = config['seq_length']

        print(f"\nTraining {exp_name}...")

        model = create_model(**config)
        start_time = time.time() # measure training time

        history = model.fit(
            data[f'X_train_{seq_len}'],
            data['y_train'],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=CALLBACKS,
            verbose=1
        )

        end_time = time.time()
        total_time = end_time - start_time
        time_per_epoch = total_time / EPOCHS

        model.save(f'../results/models/{exp_name}_model.keras') # save model and history
        trained_models[exp_name] = model
        training_histories[exp_name] = history.history
        training_times[exp_name] = time_per_epoch

        print(f"Time per epoch: {time_per_epoch:.2f} seconds")

    print("\n=== Training Complete ===")
    print(f"Total experiments trained: {len(experiments)}")

    with open('../results/training_data.pkl', 'wb') as f: # save training histories and times for evaluate.py
        pickle.dump({
            'histories': training_histories,
            'times': training_times
        }, f)
    print("Training data saved to results/training_data.pkl")