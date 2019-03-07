import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import initializers
from nltk.corpus import wordnet
trainDataPath = 'data_train.csv'
testDataPath = 'test_data.csv'

def main():
    data = pd.read_csv('C:\\Users\\SoulSold\\source\\repos\\PythonApplication4\\PythonApplication4\\data_train (1).csv', sep = '\t')
    data = data[['sentiment', 'text']]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    
    print('train data:')
    print(data.head(20))
    max_fatures = 2000
    tokenizer = Tokenizer(num_words=max_fatures,  split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X, maxlen= 300)
    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = 300))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
    batch_size = 32
    history = model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
    model.save('model.h5')
if __name__== "__main__":
    main()