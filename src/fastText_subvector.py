import sys, os, re, csv, codecs, numpy as np, pandas as pd
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import os
from utils import tokenizerTfIdf, loadDataSets, loadEmbedding
from fastText import load_model
window_length = 200 # The amount of words we look at per example. Experiment with this.

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]

    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x

def data_generator(df, batch_size, trainTest = True):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data

    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch

        for i, row in df.iterrows():
            comment = row['comment_text']

            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                if trainTest:
                    batch_y = np.zeros((batch_size, len(classes)), dtype='float32')

            batch_x[batch_i] = text_to_vector(comment)
            if trainTest:
                batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                if trainTest:
                    yield batch_x, batch_y
                else:
                    yield batch_x
                batch_x = None
                batch_y = None
                batch_i = 0


PATH_TO_MODEL = '../../wiki.en.bin'

print('\nLoading data')
train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')

classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

print('\nLoading FT model')
ft_model = load_model(PATH_TO_MODEL)
n_features = ft_model.get_dimension()

# Split the dataset:
split_index = int(round(len(train) * 0.9))
shuffled_train = train.sample(frac=1)
df_train = shuffled_train.iloc[:split_index]
df_val = shuffled_train.iloc[split_index:]

# Convert validation set to fixed array
x_val = df_to_data(df_val)
y_val = df_val[classes].values


batch_size = 128
training_steps_per_epoch = int(round(len(df_train) / batch_size))
training_generator = data_generator(df_train, batch_size, True)
test_batch_size = 1024
test_generator = data_generator(test, test_batch_size, False)
test_total_steps = int(round(test.shape[0] / test_batch_size))


# Ready to start training:
print "Building the model"
inp = Input(shape=(window_length, n_features))
x = Bidirectional(LSTM(50, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))(inp)
x = GlobalMaxPool1D()(x)
x = BatchNormalization()(x)
x = Dense(50, activation="relu")(x)
#x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)


import keras.backend as K
def loss(y_true, y_pred):
     return K.binary_crossentropy(y_true, y_pred)

opt = optimizers.Nadam(lr=0.001)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

ra_val = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
print "Starting model training"
model.fit_generator(
    training_generator,
    steps_per_epoch=training_steps_per_epoch,
    epochs = 1,
    validation_data=(x_val, y_val),
    callbacks=[ra_val]
)

y_test = model.predict_generator(test_generator, steps = test_total_steps)
timeStr = str(datetime.now().date()).replace('-', '_') + ' ' + str(datetime.now().time()).replace(':', '_').replace('.', '_')
sample_submission = pd.DataFrame(data = {"id": test.id.values})
sample_submission = pd.concat([sample_submission, pd.DataFrame(y_test, columns = classes)], axis=1)
sample_submission.to_csv('fastText-subvector-submission'+ timeStr +'.csv', index=False)
