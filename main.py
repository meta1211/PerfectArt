# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import csv

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras.models import load_model

#nltk.download('wordnet')
#nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

maxFeatures = 2000
maxVectorLen = 300
embed_dim = 128
lstm_out = 196
batch_size = 32

tokenizer = Tokenizer(num_words=maxFeatures,  split=' ')


patterns = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[\w+]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'\d+', # numbers
    r"['-]\w+", # words with - and '
    r"[:;=%x][o0\-^_]?[ds\\\[\]\(\)/i|><]+", # smiles
]



def ClearFromPatterns(str, patterns):
    result = str
    for pattern in patterns:
        result = re.sub(pattern, '', result)
    return result

def Split(text):
    splitPatter = r"[!?.,:\( \)\\/\"\'*;\[\=+&-]"
    return re.split(splitPatter,text)

def DeleteStopWords(str, stopWords):
    return ' '.join([word for word in str.split() if word not in stopWords])

def ClearText(DataColumn):
    DataColumn = DataColumn.apply(lambda x: x.lower())
    DataColumn = DataColumn.apply(lambda x: ClearFromPatterns(x, patterns))
    DataColumn = DataColumn.apply(lambda x: DeleteStopWords(x, stop_words))
    return DataColumn

def PrepareSet(set, maxVectorLen):
    tokenizer.fit_on_texts(set)
    X = tokenizer.texts_to_sequences(set)
    X = pad_sequences(X, maxlen= maxVectorLen)
    return X
#TO DO (Daniel)
def IsForeign(text):
    return False
#TO DO (Joseph)
def IsRealText(text):
    return True
#TO DO (Everyone)
def IsMissEmotional(text):
    return False

#TO DO: add missemotional and foreign language check
def CheckApplicablity(text):
    return  not IsForeign(text) and IsRealText(text) and not IsMissEmotional(text)

def main():
    #Just write your data path set will be prepared automaticly.
    dataPath = 'C:\\Users\\SoulSold\\source\\repos\\PythonApplication4\\PythonApplication4\\data_train (1).csv'
    outputPath = 'DATA.csv'

    data = pd.read_csv(dataPath, sep = '\t')
    set = ClearText(data['text']).values
    X = PrepareSet(set, maxVectorLen)
    model = load_model('model.h5')
    pr = model.predict(X, steps = 1)

    #creating output file with our analysis
    output = pd.DataFrame()
    output['text'] = data['text']
    output['applicity'] = ['1' if CheckApplicablity(text) else '0' for text in set]
    output['sentiment'] = ['0' if sentiment[0] > sentiment[1] else '1' for sentiment in pr]
    print(output.head(20))
    output.to_csv(outputPath)

if __name__== "__main__":
    main()