from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import numpy as np
embedding_dir = {
    'glove': '../glove.6B',
    'fasttext': '../fasttext'
}

class tokenizerTfIdf(object):
    def __init__(self, ngram_range, max_features = None):
        self.vec = TfidfVectorizer(ngram_range=ngram_range, min_df=3, max_df=0.9,
                                   stop_words='english', strip_accents='unicode',
                                   use_idf=1, smooth_idf=1, sublinear_tf=1,
                                   max_features=max_features)


    def fit_on_texts(self, textList):
        self.vec.fit(textList)
        self.word_index = self.vec.vocabulary_

    def texts_to_sequences(self, seqList):
        tokenized_text_list = []
        list_tfIdfScores = self.vec.transform(seqList)
        for i, li in enumerate(seqList):
            tokens = map(lambda x: self.word_index[x], filter(lambda y: self.word_index.get(y, None) is not None, li.split(' ')))
            tokens_sorted = sorted(tokens, key = lambda x: -list_tfIdfScores[i, x])
            tokenized_text_list.append(tokens_sorted)
        return tokenized_text_list


def loadDatafile(fileName):
    DATA_DIR = '../datasets'
    return pd.read_csv(os.path.join(DATA_DIR, fileName))


def loadDataSets():
    files = ['train.csv', 'test.csv', 'combinedProcessedDataSet.csv']
    return [loadDatafile(x) for x in files]

def get_coefs(word,*arr):
  return word, np.asarray(arr, dtype='float32')

def get_emb_vectors(embeddings_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    return embeddings_index, emb_mean, emb_std

def loadGlove(EMBEDFILE):
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDFILE) )
    return get_emb_vectors(embeddings_index)

def loadFastText(EMBEDFILE):
    with open(EMBEDFILE) as f:
        next(f)
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in f)
    del embeddings_index['-0.1719']
    return get_emb_vectors(embeddings_index)

def loadEmbedding(type, max_features, embed_size, tokenizer):
    EMBEDDING_DIR = embedding_dir[type]
    EMBEDDING_FILE = os.path.join(EMBEDDING_DIR, str(embed_size)+'d.txt')
    if type == 'glove':
        embeddings_index, emb_mean, emb_std = loadGlove(EMBEDDING_FILE)
    elif type == 'fasttext':
        embeddings_index, emb_mean, emb_std = loadFastText(EMBEDDING_FILE)

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
