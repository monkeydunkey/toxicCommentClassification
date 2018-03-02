import sys, os, re, csv, codecs, numpy as np, pandas as pd
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D
from keras.models import Model
from keras.layers.merge import concatenate
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback, EarlyStopping
from sklearn.model_selection import train_test_split
import os
from utils import tokenizerTfIdf, loadDataSets, loadEmbedding, setupModelRun

np.random.seed(42)

def schedule(ind):
    a = [0.002,0.003, 0.001, 0.001, 0.001]
    return a[ind]

#This is to save model output whenever we see an improvement in the model performance
class RocAucEvaluation(Callback):
    def __init__(self, modelDir, list_classes, test_data, validation_data=(),  interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.X_test = test_data
        self.modelDir = modelDir
        pd.DataFrame(self.X_val).to_csv(os.path.join(modelDir, 'validation_data_x.csv'), index=False)
        pd.DataFrame(self.y_val).to_csv(os.path.join(modelDir, 'validation_data_y.csv'), index=False)
        self.auc_history = []
        self.best_AUC_Score = float("-inf")
        self.list_classes = list_classes

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            y_pred_test = self.model.predict(self.X_test, batch_size=1024, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))

            self.auc_history.append(score)

            #Saving the model weights if the AUC score is the best observed till now
            if self.best_AUC_Score < self.auc_history[-1] and len(self.auc_history) > 1:
                dateTag = str(datetime.now().replace(second=0, microsecond=0)).replace(' ', '_').replace('-', '_').replace(':', '_')
                filepath_val = os.path.join(self.modelDir, 'predictions_val_' + str(round(self.auc_history[-1] * 100, 5)).replace('.', '_') + '.csv')
                filepath_test = os.path.join(self.modelDir, 'predictions_test_' + str(round(self.auc_history[-1] * 100, 5)).replace('.', '_') + '.csv')
                pd.DataFrame(y_pred, columns = self.list_classes).to_csv(filepath_val)
                pd.DataFrame(y_pred_test, columns = self.list_classes).to_csv(filepath_test)
            self.best_AUC_Score = self.auc_history[-1]

        return

embed_size = 300 # how big is each word vector
max_features = 30000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200 # max number of words in a comment to use

print "setting up directory"
runDir = setupModelRun('GRU')

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


embedding_matrix = loadEmbedding('fasttext', max_features, embed_size, tokenizer)
print "embeddings loaded"

print "Building the model"
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = SpatialDropout1D(0.2)(x)
#x = Bidirectional(GRU(64, return_sequences=True,dropout=0.1, recurrent_dropout=0.3))(x)
x = Bidirectional(GRU(80, return_sequences=True))(x)
x_1 = GlobalMaxPool1D()(x)
x_2 = GlobalAveragePooling1D()(x)
x = concatenate([x_1, x_2])
x = BatchNormalization()(x)
#x = Dense(50, activation="relu")(x)
#x = BatchNormalization()(x)
#x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

import keras.backend as K
def loss(y_true, y_pred):
     return K.binary_crossentropy(y_true, y_pred)

#opt = optimizers.Nadam(lr=0.001)
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

earStop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=1)

cv = 3
randomSeed = 42
batch_size = 64
epoch = 2
testBatchSize = 1024
aucHistory = []

predfull = np.zeros((test.shape[0], len(list_classes)))

for cv_ in xrange(cv):
    print 'Cross Validation Round', cv_ 
    [X_train, X_val, y_train, y_val] = train_test_split(X_t, y, train_size=0.95, random_state = randomSeed)
    
    ra_val = RocAucEvaluation(runDir, list_classes, [X_te], validation_data=(X_val, y_val), interval=1)
    print "Starting model training"
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,  validation_data=(X_val, y_val), callbacks=[earStop, ra_val])
    aucHistory.extend(ra_val.auc_history)
    y_test = model.predict([X_te], batch_size=testBatchSize, verbose=1)
    predfull += y_test
    print 'Average AUC Score on the validation set for cv round: ', cv_, ' is: ', np.array(ra_val.auc_history).mean()
    print '-'*53
    print '-'*53

print 'Average AUC Score: ', np.array(aucHistory).mean()
predfull /= cv


timeStr = str(datetime.now().date()).replace('-', '_') + ' ' + str(datetime.now().time()).replace(':', '_').replace('.', '_')
sample_submission = pd.DataFrame(data = {"id": test.id.values})
sample_submission = pd.concat([sample_submission, pd.DataFrame(predfull, columns = list_classes)], axis=1)
sample_submission.to_csv('GRU-submission'+ timeStr +'.csv', index=False)
