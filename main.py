#data processing modules
import pandas as pd 
import re
#text pre-procassing modules
import nltk
import os # no better solution for language identification is available atm
from autocorrect import spell
from nltk.corpus import stopwords, words
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
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

stop_words = set(stopwords.words('english'))
tokenizer = Tokenizer(num_words=maxFeatures,  split=' ')
with open('big_dict 0.5.txt', 'r') as file:
    english_vocab = [word.replace("\n", '') for word in file.readlines() if len(word) > 0]

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
    #splitPatter = r"[!@#$%^&*\(\)-+\[\]{}:;\'\"< >?/.,]"
    #splitPatter = r"[!?.,:\( \) \\/\"\'*;\[\=+&-]"
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
def IsForeign(text):
    # IMPORTANT! Before we fix this preposterous issue, you need to have langdetect.ftz file in your working directory
    data_for_test = pd.DataFrame({'email':data['text'], 'if_english':'none'})
    data_for_test.to_csv('/home/odduser/Desktop/z/hakaton/PerfectArt-master/Data/data_for_test.txt', header = False)
    os.system("./fasttext " "predict " "langdetect.ftz " "data_for_test.txt " "> " "results.txt ")
    test_results = pd.read_csv('results.txt', sep="\t", header=None)
    test_results.columns = ['lang_label']
    test_results['if_eng'] = 0
    j=0
    for lang in test_results.loc[ : , "lang_label"]:
        if lang == '__label__eng':
            test_results.loc[j, "if_eng"] = 1
        j = j + 1
    #print(test_results)
    j=0
    for lang in test_results.loc[ : , "if_eng"]:
        if lang == 0:
            data.loc[j, "applicability"] = 0
        j = j + 1
    return data

def IsRealText(text, threshold = 0.1):
    unusual = [word for word in text.split() if word not in english_vocab and len(word) > 0]
    return float(len(unusual))/len(text) < threshold
#TO DO (Maria)
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
    data['text'] = data['text'].apply(lambda x: GrammarPreProcessing(x))
    set = MakePreprocessData(data['text'].values)
    print(set['text'].values[0:10])
    X = PrepareSet(set['text'].values, maxVectorLen)
    model = load_model('model v.1.4.h5')
    pr = model.predict(X, steps = 1)
    #creating output file with our analysis
    output = pd.DataFrame()
    output['text'] = data['text']
    output['applicity'] = ['1' if CheckApplicablity(text) else '0' for text in set['text'].values]
    output['sentiment'] = ['0' if sentiment[0] > sentiment[1] else '1' for sentiment in pr]
    print(output.head(10))
    output.to_csv(outputPath)

if __name__== "__main__":
    main()
