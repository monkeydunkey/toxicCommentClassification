#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack
from scipy.special import logit, expit

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import tokenizerTfIdf, loadDataSets, loadEmbedding, setupModelRun
from nbsvm import nbsvm
# Functions
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()


model_type = 'lrchar'
todate = time.strftime("%d%m")

# read data
train, test, combined = loadDataSets()

id_train = train['id'].copy()
id_test = test['id'].copy()

# add empty label for None
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
# fill missing values
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


# Tf-idf
# prepare tokenizer
re_tok = re.compile(u'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])', re.UNICODE)

# create sparse matrices
n = train.shape[0]
#vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,  min_df=3, max_df=0.9, strip_accents='unicode',
#                      use_idf=1, smooth_idf=1, sublinear_tf=1 )

word_vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    min_df = 5,
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3))
#     ,
#     max_features=250000)
all1 = pd.concat([train[COMMENT], test[COMMENT]])
print 'Fitting word vectorizer'
word_vectorizer.fit(all1)
xtrain1 = word_vectorizer.transform(train[COMMENT])
xtest1 = word_vectorizer.transform(test[COMMENT])

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    min_df = 3,
    ngram_range=(1, 6))
#     ,
#     max_features=250000)

all1 = pd.concat([train[COMMENT], test[COMMENT]])
print 'Fitting char vectorizer'
char_vectorizer.fit(all1)

xtrain2 = char_vectorizer.transform(train[COMMENT])
xtest2 = char_vectorizer.transform(test[COMMENT])

print 'Vectorizer fit'
nfolds = 3
seed = 29
cval = 4

# data setup
xtrain = hstack([xtrain1, xtrain2], format='csr')
xtest = hstack([xtest1,xtest2], format='csr')
ytrain = np.array(train[label_cols].copy())

# stratified split
skf = StratifiedKFold(n_splits=nfolds, random_state=seed)

# storage structures for prval / prfull
predVal = np.zeros((xtrain.shape[0], len(label_cols)))
predTest = np.zeros((xtest.shape[0], len(label_cols)))
scoremat = np.zeros((nfolds,len(label_cols) ))
score_vec = np.zeros((len(label_cols),1))

mdl = nbsvm(dual = False, C = cval)
for (lab_ind,lab) in enumerate(label_cols):
    y = train[lab].copy()
    print('label:' + str(lab_ind))
    for (f, (train_index, test_index)) in enumerate(skf.split(xtrain, y)):
        # split
        x0, x1 = xtrain[train_index], xtrain[test_index]
        y0, y1 = y[train_index], y[test_index]
        # fit model for prval
        mdl.fit(x0, y0)
        #m,r = get_mdl(y0,x0, c0 = cval)
        predVal[test_index,lab_ind] = mdl.predict_proba(x1)[:,1]#m.predict_proba(x1.multiply(r))[:,1]
        scoremat[f,lab_ind] = roc_auc_score(y1,predVal[test_index,lab_ind])
        # fit model full
        #m,r = get_mdl(y,xtrain, c0 = cval)
        mdl.fit(x0, y0)
        predTest[:,lab_ind] += mdl.predict_proba(xtest)[:,1]
        print('fit:'+ str(lab) + ' fold:' + str(f) + ' score:%.6f' %(scoremat[f,lab_ind]))
#    break
predTest /= nfolds


score_vec = np.zeros((len(label_cols),1))
for ii in range(len(label_cols)):
    score_vec[ii] = roc_auc_score(ytrain[:,ii], predVal[:,ii])
print(score_vec.mean())
#print(multi_roc_auc_score(ymat, predVal))

# store validation predictions
prval = pd.DataFrame(predVal, columns = label_cols)
prval['id'] = id_train
prval.to_csv('prval_'+model_type+'x'+str(cval)+'f'+str(nfolds)+'_'+todate+'.csv', index= False)

# store the prediction on test set
submission = pd.DataFrame(predTest, columns = label_cols)
submission['id'] = id_test
submission.to_csv('prfull_'+model_type+'x'+str(cval)+'f'+str(nfolds)+'_'+todate+'.csv', index= False)
submission.to_csv('sub_' + model_type + 'x' + str(cval) + 'f' + str(nfolds) + '_' + todate + '.csv', index= False)
'''
# store submission
sample_submission = pd.DataFrame(data = {"id": test.id.values})
sample_submission = pd.concat([sample_submission, pd.DataFrame(predTest, columns = label_cols)], axis=1)
sample_submission.to_csv('sub_'+model_type+'x'+str(cval)+'f'+str(nfolds)+'_'+todate+'.csv', index= False)
'''
