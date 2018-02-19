import pandas as pd
import numpy as np
import string, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import nltk

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print 'Data Loaded'

lemmatizer = nltk.stem.WordNetLemmatizer()
toRemove = ['\n', '\t', '\r']
printable = set(string.printable)
def preprocess(row):    
    try:
        comment = row['comment_text']
        comment = filter(lambda x: x in printable, comment)
        for ele in toRemove:
            comment = comment.replace(ele, '')
        comment = comment.translate(None, string.punctuation)
        #Remove user mention
        comment = re.sub('@[^\s]+','',comment)
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split(" ")])
        
        words = comment.split(' ')
        upperCaseWords = filter(lambda x: x.isupper(), words)
        row['ProcessedText'] = comment
        row['wordCount'] = len(words)
        row['shouting words'] = len(upperCaseWords)
    except Exception as e:
        print '\n\n The comment could not be processed \n\n', row, e.message
        row["ProcessedText"] = "unknown"
        row["wordCount"] = 1
        row["shouting words"] = 0
    return row

combinedDf = train_df.append(test_df)
combinedDf = combinedDf.apply(preprocess, axis=1)

combinedDf.to_csv('combinedProcessedDataSet.csv', index=False)
