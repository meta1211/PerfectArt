#data processing modules
import pandas as pd 
import re
#text pre-procassing modules
import os
from autocorrect import spell
from nltk.corpus import stopwords, words
from nltk.corpus import wordnet as wn
#model modules
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

maxFeatures = 2000 #Change carefully. May decrease quality of predictions
maxVectorLen = 300 #Warning! Dont change it. Model learned with this vector size

#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('words')
#nltk.download('sentiwordnet')

projectFolder = os.path.dirname(__file__)
stop_words = set(stopwords.words('english'))
tokenizer = Tokenizer(num_words=maxFeatures,  split=' ')
sentimentWordsPath = os.path.join(projectFolder, 'Data\\SentimentWordsData.csv')
dictPath = os.path.join(projectFolder, 'Data\\big_dict 0.5.txt')
with open(dictPath, 'r') as file:
    english_vocab = [word.replace("\n", '') for word in file.readlines() if len(word) > 0]
sentimentWords = pd.read_csv(sentimentWordsPath)
sentiDict = {row['word'] : (row['NegScore'], row['PosScore']) for index, row in sentimentWords.iterrows()}

patterns =[
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
    regex = r'(\w*) '
    list1=re.findall(regex,text)
    return list1

def DeletePunctuation(text):
    return ' '.join([word for word in Split(text) if len(word) > 0])

def DeleteStopWords(words, stopWords):
    return [word for word in words if word not in stopWords]

def CorrectSpelling(words):
    text = [spell(word).lower() if len(word) > 3 else word for word in words]
    return text

def MorphyCorrection(words):
    res = []
    for word in words:
        newWord = wn.morphy(word) #Returns None if it cant change word
        if newWord:
            res.append(newWord)
        else:
            res.append(word)
    return res

def GrammarPreProcessing(text):
    text = text.lower()
    text = ClearFromPatterns(text, patterns)
    text = DeletePunctuation(text)
    words = text.split(' ')
    words = DeleteStopWords(words, stop_words)
    words = CorrectSpelling(words)
    words = MorphyCorrection(words)
    return ' '.join(words)

def UppercaseCount(text):
    count = 0
    for letter in text:
        if letter.isupper():
            count += 1
    return count

def SentimentPunctuationCount(text):
    return text.count('!')

def PrepareSet(set, maxVectorLen):
    tokenizer.fit_on_texts(set)
    X = tokenizer.texts_to_sequences(set)
    X = pad_sequences(X, maxlen= maxVectorLen)
    return X

def MakePreprocessData(texts):
    data = pd.DataFrame()
    data['text'] = texts
    data['len'] = [len(text) for text in texts]
    data['words_count'] = [len(text.split()) for text in texts]
    data['uppercase_count'] = [UppercaseCount(text) for text in texts]
    data['meanWords_count'] = [len([word for word in text if word not in stop_words]) for text in texts]
    data['sentimentPunctuation_count'] = [SentimentPunctuationCount(word) for word in texts]
    return data

def RealWordsRatio(text):
    if len(text) == 0:
        return 0
    unusual = [word for word in text.split() if word not in english_vocab and len(word) > 0]
    return 1 - float(len(unusual))/len(text)

def EmotionalWordsRatio(text):
    count = 0
    posSum = 0
    negSum = 0
    for word in text.split():
        if word in sentiDict:
            count += 1
            negSum = sentiDict[word][0]
            posSum = sentiDict[word][1]
    if count == 0:
        return (0, 0)
    else:
        return (negSum/count, posSum/count)
    

def main():
    #Just write your data path set will be prepared automaticly.
    dataPath = os.path.join(projectFolder, "Data\TestData.csv")
    outputPath = os.path.join(projectFolder, 'DATA.csv')
    data = pd.read_csv(dataPath)
    set = MakePreprocessData(data['text'].values)
    real_words_ratio = []
    emotWordsRatio = []
    for text in set['text'].values:
        text = GrammarPreProcessing(text)
        real_words_ratio.append(RealWordsRatio(text))
        emotWordsRatio.append(EmotionalWordsRatio(text))
    set['real_words_ratio'] = real_words_ratio
    set['emotWordsRatio'] = emotWordsRatio
    X = PrepareSet(set['text'].values, maxVectorLen)
    model = load_model('model v.1.4.h5')
    pr = model.predict(X, steps = 1)

    #creating output file with our analysis
    output = pd.DataFrame()
    output['text'] = data['text']
    output['applicity'] = ['1' if row['real_words_ratio'] > 0.9 and row['emotWordsRatio'][0] + row['emotWordsRatio'][1] > 0.05 else '0' for index, row in set.iterrows()]
    output['sentiment'] = ['0' if sentiment[0] > sentiment[1] else '1' for sentiment in pr]
    print(output.head(10))
    output.to_csv(outputPath)

if __name__== "__main__":
    main()
