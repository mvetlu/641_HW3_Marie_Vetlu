import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import set_seeds

set_seeds() # using the same random seeds everywhere in the repo
df = pd.read_csv('../data/IMDB Dataset.csv') # load CSV file

def preprocess(text): # function to preprocess text
    text = str(text).lower() # lowercase
    text = re.sub(r'[^a-z\s]', '', text) # remove punctuation and special characters
    return text


df['review'] = df['review'].apply(preprocess) # apply preprocessing on the dataset

train_texts = df['review'][:25000].tolist() # first 25k for training
test_texts = df['review'][25000:].tolist() # last 25k for testing

y_train = (df['sentiment'][:25000] == 'positive').astype(int).values # same for the labels
y_test = (df['sentiment'][25000:] == 'positive').astype(int).values

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>') # tokenize with top 10k words
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts) # convert train reviews to a sequence of token IDs
test_sequences = tokenizer.texts_to_sequences(test_texts) # and then test reviiews too

max_lengths = [25, 50, 100] # pad sequences for different lengths
data = {}

for max_len in max_lengths: # the loop to try all variations
    X_train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

    data[f'X_train_{max_len}'] = X_train_padded # training data padded to 25/50/100 tokens
    data[f'X_test_{max_len}'] = X_test_padded # testing data padded to 25/50/100 tokens

data['y_train'] = y_train # train labels
data['y_test'] = y_test # test labels
data['tokenizer'] = tokenizer # for later use

def get_data(): # to make data easily importable for other files
    return data

if __name__ == '__main__':
    print("Data preprocessing complete!")
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")