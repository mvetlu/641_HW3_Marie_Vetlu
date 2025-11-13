from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import itertools
from utils import set_seeds

set_seeds()  # using the same random seeds everywhere in the repo

def create_model(architecture='LSTM', activation='tanh', optimizer='adam',
                 seq_length=100, gradient_clipping=False, vocab_size=10000):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length)) # embedding layer size 100

    # first hidden layer size: 64
    if architecture == 'RNN':
        model.add(SimpleRNN(64, activation=activation, return_sequences=True))
    elif architecture == 'LSTM':
        model.add(LSTM(64, activation=activation, return_sequences=True))
    elif architecture == 'Bidirectional_LSTM':
        model.add(Bidirectional(LSTM(64, activation=activation, return_sequences=True)))

    model.add(Dropout(0.4))  # dropout 0.4 after the first layer

    # second hidden layer size 64
    if architecture == 'RNN':
        model.add(SimpleRNN(64, activation=activation))
    elif architecture == 'LSTM':
        model.add(LSTM(64, activation=activation))
    elif architecture == 'Bidirectional_LSTM':
        model.add(Bidirectional(LSTM(64, activation=activation)))

    model.add(Dropout(0.4))  # dropout 0.4 after the second layer
    model.add(Dense(1, activation='sigmoid')) # fully connected output layer with sigmoid

    # optimizer configuration
    if gradient_clipping:  # gradient clipping or not
        clip_value = 1.0
        if optimizer == 'adam':
            opt = Adam(clipvalue=clip_value)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=0.2, clipvalue=clip_value)
        elif optimizer == 'rmsprop':
            opt = RMSprop(clipvalue=clip_value)
    else:  # no strategy (no clipping)
        if optimizer == 'adam':
            opt = Adam()
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=0.2)
        elif optimizer == 'rmsprop':
            opt = RMSprop()

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) # compile with binary cross-entropy
    return model


# I have 56 experimental combinations
ARCHITECTURES = ['RNN', 'LSTM', 'Bidirectional_LSTM']
ACTIVATIONS = ['sigmoid', 'relu', 'tanh']
OPTIMIZERS = ['adam', 'rmsprop', 'sgd']
SEQ_LENGTHS = [25, 50, 100]
CLIPPING_VALUES = {True: 'clip', False: 'noclip'}


def get_exp_name(arch, activ, opt, length, clip_bool): # naming convention to identify different combinations
    arch_name = 'BiLSTM' if arch == 'Bidirectional_LSTM' else arch
    clip_name = CLIPPING_VALUES[clip_bool]
    return f"{arch_name}_{activ}_{opt}_{length}_{clip_name}"

experiments = []
BASE_CONFIG = {'activation': 'tanh', 'optimizer': 'adam', 'seq_length': 100, 'gradient_clipping': True}

# architecture x activation (3x3 = 9) with Length 100 and adam and clip
for arch, activ in itertools.product(ARCHITECTURES, ACTIVATIONS):
    exp = {**BASE_CONFIG, 'architecture': arch, 'activation': activ}
    exp['exp_name'] = get_exp_name(arch, activ, exp['optimizer'], exp['seq_length'], exp['gradient_clipping'])
    experiments.append(exp)

# architecture x sequence length (3x3 = 9) with stable tanh, adam, clip
for arch, length in itertools.product(ARCHITECTURES, [25, 50]):
    exp = {**BASE_CONFIG, 'architecture': arch, 'seq_length': length}
    exp['exp_name'] = get_exp_name(arch, exp['activation'], exp['optimizer'], length, exp['gradient_clipping'])
    experiments.append(exp)

# architecture x optimizer (3x3 = 9) with stable tanh, L=100, clip
for arch, opt in itertools.product(ARCHITECTURES, ['rmsprop', 'sgd']):
    exp = {**BASE_CONFIG, 'architecture': arch, 'optimizer': opt}
    exp['exp_name'] = get_exp_name(arch, exp['activation'], opt, exp['seq_length'], exp['gradient_clipping'])
    experiments.append(exp)

# RNN: 3 activ x 3 opt x 2 clip = 18 focus on fastest model RNN with L=100
rnn_params = itertools.product(ACTIVATIONS, OPTIMIZERS, CLIPPING_VALUES.keys())
for activ, opt, clip in rnn_params:
    arch = 'RNN'
    if arch == 'RNN' and activ == 'tanh' and opt == 'adam' and clip == True:
        continue

    exp = {**BASE_CONFIG, 'architecture': arch, 'activation': activ, 'optimizer': opt, 'gradient_clipping': clip}
    exp['exp_name'] = get_exp_name(arch, activ, opt, exp['seq_length'], clip)
    experiments.append(exp)

# LSTM/BiLSTM Combinations with L=50
lstm_bilstm_combos = itertools.product(
    ['LSTM', 'Bidirectional_LSTM'],
    ACTIVATIONS,
    OPTIMIZERS,
    [50],  # Fixed to medium length
    CLIPPING_VALUES.keys()
)

for arch, activ, opt, length, clip in lstm_bilstm_combos:
    if arch == 'LSTM' and activ == 'tanh' and opt == 'adam' and clip == True:
        continue # skip models that already were before

    exp = {'architecture': arch, 'activation': activ, 'optimizer': opt,
           'seq_length': length, 'gradient_clipping': clip}
    exp['exp_name'] = get_exp_name(arch, activ, opt, length, clip)
    experiments.append(exp)

# final cleanup for duplicates
final_experiments = []
exp_names = set()
for exp in experiments:
    if exp['exp_name'] not in exp_names:
        final_experiments.append(exp)
        exp_names.add(exp['exp_name'])

# override old experiments
experiments = final_experiments