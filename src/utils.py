from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import storage
from werkzeug import secure_filename
from werkzeug.exceptions import BadRequest
import six
embedding_dir = {
    'glove': '../glove.6B',
    'fasttext': '../fasttext'
}

class tokenizerTfIdf(object):
    def __init__(self, ngram_range, max_features = None):
        self.vec = TfidfVectorizer(ngram_range=ngram_range, min_df=4,
                                   use_idf=1, max_features=max_features)


    def fit_on_texts(self, textList):
        self.vec.fit(textList)
        self.word_index = self.vec.vocabulary_

    def texts_to_sequences(self, seqList):
        list_tfIdfScores = self.vec.transform(seqList)
        tokens = map(lambda x: tfidfmat[x,:].nonzero()[1], range(len(list_sentences_train)))
        print 'tokens generated'
        tfidfScores = map(lambda x: np.array(tfidfmat[x[0], x[1]].todense())[0], enumerate(tokens))
        '''
        for i, li in enumerate(seqList):
            tokens = map(lambda x: self.word_index[x], filter(lambda y: self.word_index.get(y, None) is not None, li.split(' ')))
            tfidfScores = map(lambda x: list_tfIdfScores[i, x], tokens)
            tokenized_text_list.append(tokens)
            tfidfScores_list.append(tfidfScores)
        '''
        return tokens, tfidfScores

    def pad_sequences(self, list_tokenized_train, tfidfScores, maxlen):
        paddedSeq = []
        for i, seq in enumerate(list_tokenized_train):
            seqLen = len(seq)
            padSeq = np.array(seq)[np.argsort(tfidfScores[i])[-min(maxlen, seqLen):]]
            padSeq = np.pad(padSeq, (max(0, maxlen - seqLen),0), 'constant', constant_values=0)
            paddedSeq.append(padSeq)
        return np.array(paddedSeq)


def _get_storage_client(PROJECT_ID):
    return storage.Client(project=PROJECT_ID)


def _check_extension(filename, allowed_extensions):
    if ('.' not in filename or
            filename.split('.').pop().lower() not in allowed_extensions):
        raise BadRequest(
            "{0} has an invalid name or extension".format(filename))


def _safe_filename(filename):
    """
    Generates a safe filename that is unlikely to collide with existing objects
    in Google Cloud Storage.
    ``filename.ext`` is transformed into ``filename-YYYY-MM-DD-HHMMSS.ext``
    """
    filename = secure_filename(filename)
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    basename, extension = filename.rsplit('.', 1)
    return "{0}-{1}.{2}".format(basename, date, extension)

def upload_folder(config):
    folder = config['runDir']
    for f in os.listdir(folder):
        if '.DS_Store' not in f:
            print 'Uploading Folder:', folder, 'file:',  f
            filePath = os.path.join(folder, f)
            print 'uploaded file can be found at', upload_file(filePath, config)

# [START upload_file]
def upload_file(filename, config):
    """
    Uploads a file to a given Cloud Storage bucket and returns the public url
    to the new object.
    """
    _check_extension(filename, config['ALLOWED_EXTENSIONS'])
    #filename = _safe_filename(filename)

    client = _get_storage_client(config['PROJECT_ID'])
    bucket = client.bucket(config['CLOUD_STORAGE_BUCKET'])
    blob = bucket.blob(filename)

    blob.upload_from_filename(filename)
    url = blob.public_url

    if isinstance(url, six.binary_type):
        url = url.decode('utf-8')
    return url

def writeToResults(config, text):
    text = str(text)
    filepath = os.path.join(config['runDir'], config['ResultFileName'])
    with open(filepath, 'a') as f:
        f.write(text + '\n')

def saveToRunDir(config, df, filename='runResult.csv'):
    filepath = os.path.join(config['runDir'], filename)
    df.to_csv(filepath, index=False)

def loadDatafile(fileName):
    DATA_DIR = '../datasets'
    return pd.read_csv(os.path.join(DATA_DIR, fileName))

def setupModelRun(config):
    modelName = config['ModelName']
    timeStr = str(datetime.now().date()).replace('-', '_') + '__' + str(datetime.now().time()).replace(':', '_').replace('.', '_')
    dirName = modelName + '_' + timeStr
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    with open(os.path.join(dirName, config['ResultFileName']), 'w') as f:
        f.write('Experiment:' + '\n' +config['Experiment'] + '\n\n')
    return dirName

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
    #embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
