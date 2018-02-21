import pandas as pd
import numpy as np
import string, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import nltk

train_df = pd.read_csv('../datasets/train.csv')
test_df = pd.read_csv('../datasets/test.csv')

print 'Data Loaded'

printable = set(string.printable)
def preprocess(row):
    try:
        s = row['comment_text']
        s = filter(lambda x: x in printable, s)
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
        row["ProcessedText"] = s
        return row

    except Exception as e:
        print '\n\n The comment could not be processed \n\n', row, e.message
        row["ProcessedText"] = "unknown"
    return row

combinedDf = train_df.append(test_df)
combinedDf = combinedDf.apply(preprocess, axis=1)

combinedDf.to_csv('../datasets/combinedProcessedDataSet.csv', index=False)
