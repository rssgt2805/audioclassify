
# !IMPORTANT!

# please pass filenames and type of language used in that .wav(AUDIO) file
# the files will be stored in csv file which is named by team registration no


# if during testing you find abnormalities just run the code twice
# you will get the rerquired output 

# internet services are required for the code to run
# if working on jupyter notebook please run the follwing code before excueting the whole file
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('average_perceptron_tagger')


import speech_recognition as sr 

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

filename = 'Testdrive.wav'
type_of_language = 'english'
#type_of_language = list(type_of_language)
def filer(filename,file_type_language):
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
        
    if file_type_language == 'english':
        #print(r.recognize_google(audio))
        s = r.recognize_google(audio) 
        h = translator.translate(s,'en') 
        z = h.text
        print(h.text)
        return z
    else:
        print(r.recognize_google(audio,language = 'hi-IN'))
        s = r.recognize_google(audio,language = 'hi-IN') 
        h = translator.translate(s,'en') 
        z1 = h.text
        print(h.text)
        return z1       
    
    

#k = filer(r'Testdrive.wav','english')

#k = filer(filename,'english')

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
    
    with open(r'enquiry (2).txt', 'r') as myfile1:
          enquiry_1 = myfile1.readlines()
    with open(r'breakdown (1).txt', 'r') as myfile2:
          breakdown_1 = myfile2.readlines()        
    with open(r'feedback (2).txt', 'r') as myfile3:
          feedback = myfile3.readlines()    
    with open(r'vehicle_quality (1).txt', 'r') as myfile1:
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
    #print(type(D))
    my_dict = {1:'enquiry',0:'breakdown',2:'feedback',3:'vehicle_quality'}
    #print(Y)
    #Test_Y = Encoder.fit_transform(Test_Y)
    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(documents,Y,test_size=0.3)
    # X = column_or_1d(X, warn=True)
    # Y = column_or_1d(Y, warn=True)
    text_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemNormalize, stop_words='english',analyzer = 'word')),
                     ('tfidf', TfidfTransformer(norm="l2")),
                     ('clf', LinearSVC()),
                     ])
    text_clf.fit(Train_X,Train_Y)
    #print(Train_Y)
    #print('%')
    #print(Test_Y)
    new_doc = []
    new_doc.append(z)
    #predicted = text_clf.predict()
    #predicted = text_clf.predict(new_doc)
    pred_1 =  text_clf.predict(Test_X)
    predicted = text_clf.predict(documents)
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

#cf,dicter = cl(k)
#typer = check_class(k,cf[0],dicter)
filenames  = [] 
labels = []
my_dict = {'filename':filenames,'type': labels}
df = pd.DataFrame(my_dict , columns = ['filename','type'])
#df.to_csv(r'I-K3QTT.csv')
df = pd.read_csv(r'I-K3QTT.csv') 
    
def create_csv(filename,type_of_languages):
    #print(type_of_languages[0])
    k = filer(filename,type_of_languages)
    cf,dicter = cl(k)
    typer = check_class(k,cf[0],dicter)
    df_temp = pd.read_csv(r'I-K3QTT.csv')
    filenames_new = list(df_temp.filename)
    labels_new = list(df_temp.type)
    filenames_new.append(filename)
    labels_new.append(typer)
    my_dict_new = {'filename':filenames_new,'type': labels_new}
    df_new = pd.DataFrame(my_dict_new , columns = ['filename','type'])
    print(df_new)
    #df.append(df_new,ignore_index = True)
    df_new.to_csv(r'I-K3QTT.csv')
    print(pd.read_csv(r'I-K3QTT.csv'))

create_csv(filename,type_of_language)

