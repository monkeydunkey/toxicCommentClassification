import pandas as pd
import numpy as np
import string, re
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss, roc_auc_score
from datetime import datetime
from scipy import sparse
## Defines NB SVM model
class nbsvm():
    def pr(self, y_i, y, data):
        p = data[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def fit(self, data, y):
        y = y.values
        #print 'y shape', y.shape
        r = sparse.csr_matrix(np.log(self.pr(1,y, data) / self.pr(0,y, data)))
        m = LogisticRegression(C=4, dual=True)
        x_nb = data.multiply(r)
        #np.multiply(r.reshape(1, len(r)), data.values)
        #print np.isnan(x_nb).any(), np.isnan(y).any(), np.isinf(x_nb).any(), np.isinf(y).any()
        self.model = (m.fit(x_nb, y), r)
        return self.model

    def predict_proba(self, data):
        m, r = self.model
        #return m.predict_proba(np.multiply(r.reshape(1, len(r)), data.values))
        return m.predict_proba(data.multiply(r))


combinedDf = pd.read_csv('combinedProcessedDataSet.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

combinedDf['none'] = 1-combinedDf[label_cols].max(axis=1)
combinedDf = combinedDf.reset_index(drop=True)
combinedDf['percentShout'] = combinedDf['shouting words'] / (combinedDf['wordCount'] + 1)
testData = combinedDf.loc[train_df.shape[0]:].copy()
trainData = combinedDf.loc[: train_df.shape[0] - 1].copy()

del combinedDf
del train_df
del test_df

vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, stop_words='english',
                      strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(trainData['ProcessedText'])
test_term_doc = vec.transform(testData['ProcessedText'].fillna("unknown"))

colsCopy = ['wordCount', 'shouting words', 'percentShout']
#for col in colsCopy:
    #trn_term_doc = sparse.hstack((trn_term_doc,trainData[col].values[:,None])).A
    #test_term_doc = sparse.hstack((test_term_doc,testData[col].values[:,None])).A

#trn_term_doc = sparse.csr_matrix(trn_term_doc.values)
#test_term_doc = sparse.csr_matrix(test_term_doc.values)

models = [('nbsvm', nbsvm())]#, ('extraTreeClassifier', ExtraTreesClassifier(n_jobs=-1, random_state=3))]
for mdlName, mdl in models:
    preds = np.zeros((test_term_doc.shape[0], len(label_cols)))
    print('Model Name', mdlName)
    for i, j in enumerate(label_cols):
        print('fit', j)
        mdl.fit(trn_term_doc, trainData[j].reset_index(drop=True))
        preds[:,i] = mdl.predict_proba(test_term_doc)[:,1]
    submid = pd.DataFrame({'id': map(lambda x: str(x), testData["id"].values)})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
    #print submission.shape, submission.dtypes
    timeStr = str(datetime.now().date()).replace('-', '_') + ' ' + str(datetime.now().time()).replace(':', '_').replace('.', '_')
    submission.to_csv(mdlName + '_shoutingwords' + timeStr + '.csv', index=False)
