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

def schedule(ind):
    a = [0.002,0.003, 0.000]
    return a[ind]



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

embed_size = 300 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

train, test, combined = loadDataSets()
print "Data Loaded"


list_sentences_train = combined.loc[:train.shape[0] - 1]["ProcessedText"].fillna("_na_").values
print "train sentence list", len(list_sentences_train), "train shape", train.shape[0]
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = combined.loc[train.shape[0]:]["ProcessedText"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
#tokenizer = tokenizerTfIdf(ngram_range = (1,2), max_features = max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
print "tokensFitted"
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


embedding_matrix = loadEmbedding('glove', max_features, embed_size, tokenizer)
print "embeddings loaded"

print "Building the model"
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = Bidirectional(GRU(50, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))(x)
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

model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])


lr = callbacks.LearningRateScheduler(schedule)
[X_train, X_val, y_train, y_val] = train_test_split(X_t, y, train_size=0.95)

ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
print "Starting model training"
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val), callbacks=[lr, ra_val])

y_test = model.predict([X_te], batch_size=1024, verbose=1)
timeStr = str(datetime.now().date()).replace('-', '_') + ' ' + str(datetime.now().time()).replace(':', '_').replace('.', '_')
sample_submission = pd.DataFrame(data = {"id": test.id.values})
sample_submission[list_classes] = y_test
sample_submission.to_csv('GRU-submission'+ timeStr +'.csv', index=False)
