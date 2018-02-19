from sklearn.feature_extraction.text import TfidfVectorizer

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
