
# -*- coding: utf-8 -*-

#import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams, NaiveBayesClassifier, classify, RegexpParser, ConfusionMatrix,ConditionalFreqDist
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import Word2Vec
import scipy as sp
import spacy

import os
import collections
import string
from pickle import dump
from pickle import load
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from statistics import mean
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn import svm
#from transformers import BertTokenizer

stopwords_spanish = stopwords.words('spanish')
stopwords_spanish.extend(['----','---','--',':',',','!','/','.','?','"','>'])
stopwords_spanish.extend(['…','(',')','“','”',"''",'``','•',';'])

def genera_archivo_txt(lista,nombre):
    fecha = datetime.today().strftime('%Y-%m-%d')
    news_df=pd.DataFrame(lista)
    nombreArchivo = nombre + '_' + fecha +".csv"
    news_df.to_csv(nombreArchivo, encoding='utf-8')
    
###############################################################################
#CLASIFICACIÓN EN FUNCIONAL Y NO FUNCIONAL
###############################################################################  

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
def decode_review(text,reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
  
def guardar_modelo(model,nombre):
    model_json = model.to_json()
    with open(nombre+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(nombre+".h5")
        
def cargar_modelo(nombre): 
    json_file = open(nombre+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(nombre+".h5")
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    return loaded_model

def ConvolutionalLayer(documento,test_size,learning_rate,dropout,num_epochs,filtersmap):
    '''Función para clasificar entre F/NF usando CNN'''
    vocab_size = 10000
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"  
    
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    dataframe2 = dataframe.sample(frac=1)    
    X = dataframe2['REQUISITO'].values
    y = dataframe2['ETIQUETAFINAL'].values
        
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X)    
    sequences = tokenizer.texts_to_sequences(X)    
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)  
    
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(y)        
    word_index = label_tokenizer.word_index
    print(word_index)
    labels_final = np.array(label_tokenizer.texts_to_sequences(y))
    
    X_train, X_test, y_train, y_test = train_test_split(padded, labels_final, test_size=test_size, random_state=0)

    #https://medium.com/voice-tech-podcast/text-classification-using-cnn-9ade8155dfb9
    #https://www.davidsbatista.net/blog/2018/03/31/SentenceClassificationConvNets/
    model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64),
            tf.keras.layers.Conv1D(filtersmap, 5, activation='relu'),
            tf.keras.layers.Dropout(dropout), #desactiva el 75% de las conexiones entre las neuronas
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')
    ])
        
    model.summary()    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)
        
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
        
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(train_acc,test_acc)
    print(train_loss,test_loss)
        
    testing_pred = model.predict_classes(X_test, verbose=0)
    accuracy = accuracy_score(y_test, testing_pred)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(y_test, testing_pred)
    print('Precision: %f' % precision)
    recall = recall_score(y_test, testing_pred)
    print('Recall: %f' % recall)
    f1 = f1_score(y_test, testing_pred)
    print('F1 score: %f' % f1)
    
    cr = classification_report(y_test, testing_pred, output_dict=True)
    #print(y_test)
    print('F precision:', round(cr['2']['precision'], 2))
    print('F recall:', round(cr['2']['recall'], 2))
    print('F F-measure:', round(cr['2']['f1-score'], 2))
    print('NF precision:', round(cr['1']['precision'], 2))
    print('NF recall:', round(cr['1']['recall'], 2))
    print('NF F-measure:', round(cr['1']['f1-score'], 2))
    #guardar_modelo(model,'Modelo_Convolucional_F_NF_3')

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#ConvolutionalLayer(documento,0.2,0.001,0.75,40,100) #test-size 20%,learning rate 0.001, dropout 0.75, epochs 35, filtermaps 100
    
def ConvolutionalLayer_kfold(kfold,documento,learning_rate,dropout,num_epochs,filtersmap):
    '''Función para clasificar entre F/NF usando CNN y k-fold'''
    vocab_size = 10000
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"  
    accuracy_list=[]
    f_precision=[]
    f_recall=[]
    f_f_score=[]
    fn_precision=[]
    fn_recall=[]
    fn_f_score=[]
    
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    dataframe2 = dataframe.sample(frac=1)    
    X = dataframe2['REQUISITO'].values
    y = dataframe2['ETIQUETAFINAL'].values
    
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X)    
    sequences = tokenizer.texts_to_sequences(X)    
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)  
        
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(y)        
    word_index = label_tokenizer.word_index
    print(word_index)
    labels_final = np.array(label_tokenizer.texts_to_sequences(y))
    
    n_folds = kfold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train, test in kf.split(padded):     
        
        X_train, X_test = padded[train], padded[test]
        y_train, y_test = labels_final[train], labels_final[test]
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 64),
            tf.keras.layers.Conv1D(filtersmap, 5, activation='relu'),
            tf.keras.layers.Dropout(dropout), 
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])    
        model.summary()    
        
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
     
        #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
        history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)
        
        plot_graphs(history, 'accuracy')
        plot_graphs(history, 'loss')
        
        testing_pred = model.predict_classes(X_test, verbose=0)
        accuracy = accuracy_score(y_test, testing_pred)
        accuracy_list.append(accuracy)
#        precision = precision_score(y_test, testing_pred)
#        f_precision.append(precision)
#        recall = recall_score(y_test, testing_pred)
#        f_recall.append(recall)
#        f1 = f1_score(y_test, testing_pred)
#        f_f_score.append(f1)
        cr = classification_report(y_test, testing_pred, output_dict=True)
        f_precision.append(round(cr['2']['precision'], 2))
        f_recall.append(round(cr['2']['recall'], 2))
        f_f_score.append(round(cr['2']['f1-score'], 2))
        fn_precision.append(round(cr['1']['precision'], 2))
        fn_recall.append(round(cr['1']['recall'], 2))
        fn_f_score.append(round(cr['1']['f1-score'], 2))
        
    print(np.mean(accuracy_list))   
    print(np.mean(f_precision))   
    print(np.mean(f_recall))   
    print(np.mean(f_f_score))  
    print(np.mean(fn_precision))   
    print(np.mean(fn_recall))   
    print(np.mean(fn_f_score))  
     
documento = r"datos_formal_pruebas_grupo3.csv"
ConvolutionalLayer_kfold(5,documento,0.001,0.75,40,100) #test-size 20%,learning rate 0.001, dropout 0.75, epochs 35, filtermaps 100
  
