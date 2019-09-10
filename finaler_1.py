import speech_recognition as sr 
import pyaudio
import wave

import nltk,string,numpy,math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


from googletrans import Translator


import wave
import math

import nltk,string,numpy,math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
translator = Translator()


def filer(filename):
    AUDIO_FILE = (filename) 
    r = sr.Recognizer() 

    with sr.AudioFile(AUDIO_FILE) as source: 
        audio = r.record(source) 

    try: 
        print("The audio file contains: " + r.recognize_google(audio)) 


    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio") 

    except sr.RequestError as e: 
        print(e) 
    print(r.recognize_google(audio,language = 'hi-IN'),)
    s=r.recognize_google(audio) 
    h = translator.translate(s,'en') 
    z = h.text
    print(h.text)
    return z

k = filer(r'C:\Users\user\Desktop\last\new_car.wav')



lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
     return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
     return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    
def stemming_tokenizer(words):
    #words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words_1 = [porter_stemmer.stem(word) for word in words]
    return words_1    
    
def tot_tokenizer(text):
    return textblob_tokenizer(text,LemNormalize(text))

def cosine(vect1,vect2):
    num=0
    den1=0
    den2=0
    for i in range(len(vect1)):
        num=num+vect1[i]*vect2[i]
        den1+=vect1[i]*vect1[i]
        den2+=vect2[i]*vect2[i]
    cos=num/(math.sqrt(den1)*math.sqrt(den2))    
    return(cos)


def cl(z):
    
    with open(r'C:\Users\user\Desktop\last\enquiry.txt', 'r') as myfile1:
          enquiry_1 = myfile1.readlines()
    with open(r'C:\Users\user\Desktop\last\breakdown.txt', 'r') as myfile2:
          breakdown_1 = myfile2.readlines()        
    with open(r'C:\Users\user\Desktop\last\feedback.txt', 'r') as myfile3:
          feedback = myfile3.readlines()    
    with open(r'C:\Users\user\Desktop\last\vehicle_quality.txt', 'r') as myfile1:
           vehicle_quality = myfile1.readlines()
    documents = []
    target = []
    for i in enquiry_1:
        documents.append(i)
        target.append('enquiry')
    for j in breakdown_1:
        documents.append(j)
        target.append('breakdown')
    for k in  feedback:
        documents.append(k)
        target.append('feedback')
    for i in vehicle_quality:
        documents.append(i)
        target.append('vehicle_qualiity')
    Encoder = LabelEncoder()
    Y = Encoder.fit_transform(target)
    D = list(Y)
    Y = np.array(Y).reshape(-1, 1) 
    print(type(D))
    my_dict = {1:'enquiry',0:'breakdown',2:'feedback',3:'vehicle_quality'}
    print(Y)
    #Test_Y = Encoder.fit_transform(Test_Y)
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(documents,Y,test_size=0.3)
    # X = column_or_1d(X, warn=True)
    # Y = column_or_1d(Y, warn=True)
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemNormalize, stop_words='english',analyzer = 'word')),
                     ('tfidf', TfidfTransformer(norm="l2")),
                     ('clf', LinearSVC()),
                     ])
    text_clf.fit(Train_X,Train_Y)
    print(Train_Y)
    print('%')
    print(Test_Y)
    new_doc = []
    new_doc.append(z)
    #predicted = text_clf.predict()
    predicted = text_clf.predict(new_doc)
    pred_1 =  text_clf.predict(Test_X)
    #predicted = text_clf.predict(documents)
    print(pred_1,predicted)
    print(metrics.classification_report(Test_Y, pred_1))
    #print(confusion_matrix(Test_Y, pred_1))
    print("SVM Accuracy Score -> ",accuracy_score(pred_1,Test_Y)*100)
    return predicted,my_dict




def check_class(z,predicted_class,class_dict):
    label = class_dict[predicted_class]
    if z.lower().find('test drive') != -1:
        print('class = test_drive')
        j = 'test_drive'
        return j    
    elif z.lower().find('complain')  != -1:
        print('class = vehicle_quality ')
        f = 'vehicle_quality '
        return f 
    else :
        print('class:' + label )
        return label


cf,dicter = cl(k)
typer = check_class(k,cf[0],dicter)
