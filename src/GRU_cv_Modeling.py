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
from utils import tokenizerTfIdf, loadDataSets, loadEmbedding, setupModelRun, writeToResults, saveToRunDir, upload_folder


np.random.seed(42)

def schedule(ind):
    a = [0.002,0.003, 0.001, 0.001, 0.001]
    return a[ind]

#This is to save model output whenever we see an improvement in the model performance
class RocAucEvaluation(Callback):
    def __init__(self, config, list_classes, test_data, validation_data=(),  interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.X_test = test_data
        self.config = config
        saveToRunDir(config, pd.DataFrame(self.X_val), 'validation_data_x.csv')
        saveToRunDir(config, pd.DataFrame(self.y_val), 'validation_data_y.csv')
        self.auc_history = []
        self.best_AUC_Score = float("-inf")
        self.updateStr = 'ROC-AUC - epoch: {:d} - score: {:.6f}'
        self.list_classes = list_classes

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            y_pred_test = self.model.predict(self.X_test, batch_size=1024, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            updateStr = self.updateStr.format(epoch, score)
            print(updateStr)
            writeToResults(self.config, updateStr)
            self.auc_history.append(score)
            #Saving the model weights if the AUC score is the best observed till now
            if self.best_AUC_Score < self.auc_history[-1] and len(self.auc_history) > 1:
                filepath_val = 'predictions_val_' + str(round(self.auc_history[-1] * 100, 5)).replace('.', '_') + '.csv'
                filepath_test = 'predictions_test_' + str(round(self.auc_history[-1] * 100, 5)).replace('.', '_') + '.csv'
                saveToRunDir(self.config, pd.DataFrame(y_pred, columns = self.list_classes), filepath_val)
                saveToRunDir(self.config, pd.DataFrame(y_pred_test, columns = self.list_classes), filepath_test)
            self.best_AUC_Score = self.auc_history[-1]

        return


config = {
    'embed_size': 300,# how big is each word vector
    'max_features': 30000, # how many unique words to use (i.e num rows in embedding vector)
    'maxlen': 100, # max number of words in a comment to use
    'ModelName': 'GRU',
    'Experiment': 'Checking if using tf-idf scores for choosing the 100 words for comment representation helps or not',
    'PROJECT_ID': 'experiment-168900',
    'CLOUD_STORAGE_BUCKET': 'toxiccommentclassification-experimentresults',
    'ALLOWED_EXTENSIONS': ['txt', 'csv'],
    'ResultFileName': 'results.txt'
}

print "setting up directory"
runDir = setupModelRun(config)
config['runDir'] = runDir

try:
    train, test, combined = loadDataSets()
    print "Data Loaded"

    list_sentences_train = combined.loc[:train.shape[0] - 1]["ProcessedText"].fillna("_na_").values
    print "train sentence list", len(list_sentences_train), "train shape", train.shape[0]
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = combined.loc[train.shape[0]:]["ProcessedText"].fillna("_na_").values

    #tokenizer = Tokenizer(num_words=max_features)
    tokenizer = tokenizerTfIdf(ngram_range = (1,1), max_features = config['max_features'])
    tokenizer.fit_on_texts(list(list_sentences_train))
    print "tokensFitted"
    list_tokenized_train, tfidfScores_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test, tfidfScores_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_t = tokenizer.pad_sequences(list_tokenized_train, tfidfScores_train, maxlen=config['maxlen'])
    X_te = tokenizer.pad_sequences(list_tokenized_test, tfidfScores_test, maxlen=config['maxlen'])


    embedding_matrix = loadEmbedding('fasttext', config['max_features'], config['embed_size'], tokenizer)
    print "embeddings loaded"

    print "Building the model"
    inp = Input(shape=(config['maxlen'],))
    x = Embedding(config['max_features'], config['embed_size'], weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.2)(x)
    #x = Bidirectional(GRU(64, return_sequences=True,dropout=0.1, recurrent_dropout=0.3))(x)
    #x = Bidirectional(GRU(80, return_sequences=True, recurrent_dropout=0.3))(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    x_1 = GlobalMaxPool1D()(x)
    x_2 = GlobalAveragePooling1D()(x)
    x = concatenate([x_1, x_2])
    #x = BatchNormalization()(x)
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

    cv = 1
    randomSeed = 233
    batch_size = 32
    epoch = 2
    testBatchSize = 1024
    aucHistory = []

    predfull = np.zeros((test.shape[0], len(list_classes)))

    for cv_ in xrange(cv):
        print 'Cross Validation Round', cv_
        writeToResults(config, 'Cross Validation Round: ' + str(cv_))
        [X_train, X_val, y_train, y_val] = train_test_split(X_t, y, train_size=0.95, random_state = randomSeed)

        ra_val = RocAucEvaluation(config, list_classes, [X_te], validation_data=(X_val, y_val), interval=1)
        print "Starting model training"
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,  validation_data=(X_val, y_val), callbacks=[earStop, ra_val])
        aucHistory.extend(ra_val.auc_history)
        y_test = model.predict([X_te], batch_size=testBatchSize, verbose=1)
        predfull += y_test
        updateStr = 'Average AUC Score on the validation set for cv round: '+ str(cv_)+ ' is: '+ str(np.array(ra_val.auc_history).mean())
        print updateStr
        print '-'*53
        print '-'*53
        writeToResults(config, updateStr)

    print 'Average AUC Score: ', np.array(aucHistory).mean()
    writeToResults(config, 'Average AUC Score: '+ str(np.array(aucHistory).mean()))
    predfull /= cv


    #timeStr = str(datetime.now().date()).replace('-', '_') + ' ' + str(datetime.now().time()).replace(':', '_').replace('.', '_')
    sample_submission = pd.DataFrame(data = {"id": test.id.values})
    sample_submission = pd.concat([sample_submission, pd.DataFrame(predfull, columns = list_classes)], axis=1)
    saveToRunDir(config, sample_submission, config['ModelName']+'-submission.csv')
    #sample_submission.to_csv('GRU-submission'+ timeStr +'.csv', index=False)

except Exception as e:
    raise
    print e
    writeToResults(config, e)
finally:
    # Performing final clean up and push to cloud storage
    print ('uploading run directory to google cloud storage')
    upload_folder(config)
