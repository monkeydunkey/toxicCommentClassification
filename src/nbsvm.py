from scipy import sparse
from sklearn.linear_model import LogisticRegression
import numpy as np

class nbsvm():
    def __init__(self, dual=True, C=4.0, class_weight=None):
        self.class_weight = class_weight
        self.C = C
        self.dual = dual

    def pr(self, y_i, y, data):
        p = data[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)
    def fit(self, data, y):
        y = y.values
        #print 'y shape', y.shape
        r = sparse.csr_matrix(np.log(self.pr(1,y, data) / self.pr(0,y, data)))
        m = LogisticRegression(C=self.C, dual=self.dual, class_weight = self.class_weight)
        x_nb = data.multiply(r)
        #np.multiply(r.reshape(1, len(r)), data.values)
        #print np.isnan(x_nb).any(), np.isnan(y).any(), np.isinf(x_nb).any(), np.isinf(y).any()
        self.model = (m.fit(x_nb, y), r)
        return self.model
    def predict_proba(self, data):
        m, r = self.model
        #return m.predict_proba(np.multiply(r.reshape(1, len(r)), data.values))
        return m.predict_proba(data.multiply(r))

    def predict(self, data):
        proba = self.predict_proba(data)
        #return m.predict_proba(np.multiply(r.reshape(1, len(r)), data.values))
        return (proba[:, 1] > 0.5).astype(int)
