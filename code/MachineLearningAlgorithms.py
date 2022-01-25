
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

def get_features_for_class(text):
    features = {}
    for word in word_tokenize(text):
        features['contains({})'.format(word.lower())] = True
    return features

def clean_words(words,stopwords):
    words_clean = []    
    for word in word_tokenize(words):
        word = word.lower()
        if word not in stopwords and word not in string.punctuation:
            words_clean.append(stemmer.stem(word))
    return words_clean
  
def bag_of_words(words):
    words_clean = clean_words(words,stopwords_spanish)       
    dict1={}        
    for word in words_clean:
        val = True
        dict1[word] = val     
    return dict1

def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)    
    return words_dictionary

def bag_of_all_words(words, n=2):    
    unigram_features = bag_of_words(words)
    all_features = unigram_features.copy()
    
    important_words = ['arriba', 'abajo', 'sobre', 'debajo', 'mas', 'mucho',  'no', 'solo', 'que', 'demasiado', 'muy', 'just', 'pero']
    stopwords_spanish_for_bigrams = set(stopwords_spanish) - set(important_words)
    words_clean_for_bigrams = clean_words(words, stopwords_spanish_for_bigrams)
    
    for i in range(2,n+1):
        feature_gram = bag_of_ngrams(words_clean_for_bigrams,i)
        all_features.update(feature_gram)  
    return all_features

#print(bag_of_all_words('esta es prueba de texto para bigramas y trigramas', 2))
#print(bag_of_all_words('esta es prueba de texto para bigramas y trigramas', 3))

def tokenize(words):
    words_clean = []    
    for word in word_tokenize(words):
        word = word.lower()
        if word not in string.punctuation and word not in stopwords_spanish:
            words_clean.append(stemmer.stem(word))
    return words_clean

def tf_idf_Vec(X1,ngram,tipo):
    
    if tipo == 0:        #stemmer
        if ngram > 0:
            tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize,analyzer='word',
                            max_features=1000,ngram_range =(1,ngram),use_idf=True) 
        else:        
            tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize,analyzer='word',
                            max_features=1000,use_idf=True) 
    else:        
        if ngram > 0:
            tfidf_vectorizer = TfidfVectorizer(analyzer='word',stop_words=stopwords_spanish,
                            max_features=1000,ngram_range =(1,ngram),use_idf=True) 
        else:        
            tfidf_vectorizer = TfidfVectorizer(analyzer='word',stop_words=stopwords_spanish,
                            max_features=1000,use_idf=True) 
    
    X = tfidf_vectorizer.fit_transform(X1).toarray()
    return X,tfidf_vectorizer

def tf_idf_Vec2(X1,tfidf_vectorizer):    
    X = tfidf_vectorizer.transform(X1).toarray()
    return X

def NaiveBayes(documento, tipo, ngram, test_size,graba):    
    accuracy_list=[]
    train_set=[]
    test_set=[]
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    X = dataframe['REQUISITO'].values.tolist()
    y = dataframe['ETIQUETAFINAL'].values.tolist()
    print(y)
    
    print(len(X))
    print(len(y))
    if tipo == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        print(len(X_train))
        print(len(X_test))
        for i in range(0,len(X_train)):
            train_set.append([bag_of_words(X_train[i]), y_train[i]]) 
        for i in range(0,len(X_test)):
            test_set.append([bag_of_words(X_test[i]), y_test[i]])
            
    if tipo == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=0)
        for i in range(0,len(X_train)):
            train_set.append([bag_of_all_words(X_train[i],ngram), y_train[i]]) 
        for i in range(0,len(X_test)):
            test_set.append([bag_of_all_words(X_test[i],ngram), y_test[i]])
      
    print(len(train_set), len(test_set))
    #print(train_set,test_set)  
    #print(train_set.shape)
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy_list.append(classify.accuracy(classifier, test_set))
    print('Naive Bayes classifier',classify.accuracy(classifier, test_set)) 
    y_test_ = []
    y_pred_ = []
        
    for i, (feats, label) in enumerate(test_set):
        #print(i, feats, label)
        observed = classifier.classify(feats)
        y_test_.append(label)
        y_pred_.append(observed)
    
    cr = classification_report(y_test_, y_pred_)
    print(cr)
    cr = classification_report(y_test_, y_pred_, output_dict=True)
    print(cr)
    print('F precision:', round(cr['F']['precision'], 2))
    print('F recall:', round(cr['F']['recall'], 2))
    print('F F-measure:', round(cr['F']['f1-score'], 2))
    #print('F accuracy:', round(cr['F']['accuracy'], 2))
    print('NF precision:', round(cr['NF']['precision'], 2))
    print('NF recall:', round(cr['NF']['recall'], 2))
    print('NF F-measure:', round(cr['NF']['f1-score'], 2))
    #print('NF accuracy:', round(cr['NF']['accuracy'], 2))
    
    if graba == 1:
        nombre = 'naive_BOW'+'_'+str(ngram)+'.pkl'
        #output = open('class_naive.pkl',  'wb')
        output = open(nombre,  'wb')
        dump(classifier,  output, -1)
        output.close()

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#NaiveBayes(documento,0,0,0.20,0) #Bag of Word
#NaiveBayes(documento,1,2,0.20,0) #Bag of Word Bigram
#NaiveBayes(documento,1,3,0.20,0) #Bag of Word Trigram

def NaiveBayes_K_fold(documento, tipo, ngram, num_folds):    
    accuracy_list=[]
    f_precision=[]
    f_recall=[]
    f_f_score=[]
    nf_precision=[]
    nf_recall=[]
    nf_f_score=[]
    
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    dataframe2 = dataframe.sample(frac=1)
    X = dataframe2['REQUISITO'].values.tolist()
    y = dataframe2['ETIQUETAFINAL'].values.tolist()
    print(y)
    
    subset_size = round(len(X)/num_folds)
    print(len(X))
    print(subset_size)
    
    for k in range(1,num_folds):
        train_set=[]
        test_set=[]
        X_train = X[:k*subset_size] + X[(k + 1)*subset_size:]
        y_train = y[:k*subset_size] + y[(k + 1)*subset_size:]
        X_test = X[k*subset_size:][:subset_size]
        y_test = y[k*subset_size:][:subset_size]        
        
        for i in range(0,len(X_train)):
            if tipo == 0:
                train_set.append([bag_of_words(X_train[i]), y_train[i]]) 
            else:
                train_set.append([bag_of_all_words(X_train[i],ngram), y_train[i]]) 
        for i in range(0,len(X_test)):
            if tipo == 0:
                test_set.append([bag_of_words(X_test[i]), y_test[i]])
            else:
                test_set.append([bag_of_all_words(X_test[i],ngram), y_test[i]])
      
        classifier = NaiveBayesClassifier.train(train_set)
        accuracy_list.append(classify.accuracy(classifier, test_set))
        print('Naive Bayes classifier',classify.accuracy(classifier, test_set)) 
        print(len(train_set), len(test_set))
        y_test_ = []
        y_pred_ = []
        
        for i, (feats, label) in enumerate(test_set):
           observed = classifier.classify(feats)
           y_test_.append(label)
           y_pred_.append(observed)
        
        cr = classification_report(y_test_, y_pred_, output_dict=True)        
        f_precision.append(round(cr['F']['precision'], 2))
        f_recall.append(round(cr['F']['recall'], 2))
        f_f_score.append(round(cr['F']['f1-score'], 2))
        nf_precision.append(round(cr['NF']['precision'], 2))
        nf_recall.append(round(cr['NF']['recall'], 2))
        nf_f_score.append(round(cr['NF']['f1-score'], 2))        
        
    print('accuracy', round(mean(accuracy_list), 2))
    print(accuracy_list)
    print('F precision:', round(mean(f_precision), 2))
    print('F recall:', round(mean(f_recall), 2))
    print('F F-measure:', round(mean(f_f_score), 2))
    print('NF precision:', round(mean(nf_precision), 2))
    print('NF recall:', round(mean(nf_recall), 2))
    print('NF F-measure:', round(mean(nf_f_score), 2))

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#NaiveBayes_K_fold(documento,0,0,10) #Bag of Word
#NaiveBayes_K_fold(documento,1,2,10) #Bag of Word Bigram
#NaiveBayes_K_fold(documento,1,3,10) #Bag of Word Trigram        
  
def bag_of_words_CountVec(X1,ngram,tipo):
    
    if tipo == 0:        #tokenize usa stopwords y stemmer
        if ngram > 0:
            matrix_vectorizer = CountVectorizer(tokenizer=tokenize,analyzer='word',
                            max_features=1000,ngram_range =(1,ngram)) 
        else:        
            matrix_vectorizer = CountVectorizer(tokenizer=tokenize,analyzer='word',
                            max_features=1000) 
    else:        
        if ngram > 0:
            matrix_vectorizer = CountVectorizer(analyzer='word',stop_words=stopwords_spanish,
                            max_features=1000,ngram_range =(1,ngram)) 
        else:        
            matrix_vectorizer = CountVectorizer(analyzer='word',stop_words=stopwords_spanish,
                            max_features=1000) 
    
    #X = matrix_vectorizer.fit_transform(X1).toarray()  
    X1 = matrix_vectorizer.fit_transform(X1)
    X = X1.toarray()  
    return X,matrix_vectorizer

def bag_of_words_CountVec2(X1,count_vectorizer):    
    X = count_vectorizer.transform(X1).toarray()  
    return X

def GaussianNaiveBayes(documento, tipo, ngram, test_size, graba):
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    X1 = dataframe['REQUISITO'].to_numpy()
    y = dataframe['ETIQUETAFINAL'].to_numpy()
    
    if tipo == 0:
        (X,count_vectorizer) = bag_of_words_CountVec(X1,ngram,0)
    if tipo == 1:
        (X,tfidf_vectorizer) = tf_idf_Vec(X1,ngram,0)    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=0)
    print(len(X_train))
    print(len(X_test))
    print(y_test)
    
    gnb = GaussianNB()
    classifier = gnb.fit(X_train, y_train)    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    #cm = confusion_matrix(y_test, y_pred)
    #print(cm)
    cr = classification_report(y_test, y_pred, output_dict=True)
    print(cr)
    print('F precision:', round(cr['F']['precision'], 2))
    print('F recall:', round(cr['F']['recall'], 2))
    print('F F-measure:', round(cr['F']['f1-score'], 2))
    print('NF precision:', round(cr['NF']['precision'], 2))
    print('NF recall:', round(cr['NF']['recall'], 2))
    print('NF F-measure:', round(cr['NF']['f1-score'], 2))
    
    if graba == 1:
        if tipo == 0:
            nombre = 'gaussian_class_naive_BOW_'+str(ngram)                
            output = open(nombre+'_count_vectorizer.pkl',  'wb')
            dump(count_vectorizer,  output, -1)
            output.close()
    
        if tipo == 1:
            nombre = 'gaussian_class_naive_TFIDF_'+str(ngram)               
            output = open(nombre+'_tfidf_vectorizer.pkl',  'wb')
            dump(tfidf_vectorizer,  output, -1)
            output.close()
            
        output2 = open(nombre+'.pkl',  'wb')
        dump(classifier,  output2, -1)
        output2.close()

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#GaussianNaiveBayes(documento,0,0,0.20,0) #BOW CountVectorizer
#GaussianNaiveBayes(documento,0,2,0.20,0) #BOW CountVectorizer Bigram
#GaussianNaiveBayes(documento,0,3,0.20,0) #BOW CountVectorizer Trigram
#GaussianNaiveBayes(documento,1,0,0.20,0) #TfidfVectorizer
#GaussianNaiveBayes(documento,1,2,0.20,0) #TfidfVectorizer Bigram
#GaussianNaiveBayes(documento,1,3,0.20,0) #TfidfVectorizer Trigram
    
def GaussianNaiveBayes_k_fold(documento,tipo, ngram):
    accuracy_list=[]
    f_precision=[]
    f_recall=[]
    f_f_score=[]
    nf_precision=[]
    nf_recall=[]
    nf_f_score=[]
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    dataframe2 = dataframe.sample(frac=1)    
    X1 = dataframe2['REQUISITO'].to_numpy()
    y = dataframe2['ETIQUETAFINAL'].to_numpy()
    
    if tipo == 0:
        (X,count_vectorizer) = bag_of_words_CountVec(X1,ngram,0)
    if tipo == 1:
        (X,tfidf_vectorizer) = tf_idf_Vec(X1,ngram,0) 
        
    n_folds = 10
    print(len(X1))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        print(len(X_train),len(X_test))
        gnb = GaussianNB()
        classifier = gnb.fit(X_train, y_train)    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        accuracy_list.append(accuracy)
        cr = classification_report(y_test, y_pred, output_dict=True)    
        f_precision.append(cr['F']['precision'])
        f_recall.append(cr['F']['recall'])
        f_f_score.append(cr['F']['f1-score'])
        nf_precision.append(cr['NF']['precision'])
        nf_recall.append(cr['NF']['recall'])
        nf_f_score.append(cr['NF']['f1-score'])
    
    print('accuracy average', round(mean(accuracy_list), 2))
    print(accuracy_list)
    print('F precision:', round(mean(f_precision), 2))
    print('F recall:', round(mean(f_recall), 2))
    print('F F-measure:', round(mean(f_f_score), 2))
    print('NF precision:', round(mean(nf_precision), 2))
    print('NF recall:', round(mean(nf_recall), 2))
    print('NF F-measure:', round(mean(nf_f_score), 2))
    
#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#GaussianNaiveBayes_k_fold(documento,0,0) #BOW CountVectorizer
#GaussianNaiveBayes_k_fold(documento,0,2) #BOW CountVectorizer Bigram
#GaussianNaiveBayes_k_fold(documento,0,3) #BOW CountVectorizer Trigram
#GaussianNaiveBayes_k_fold(documento,1,0) #TfidfVectorizer
#GaussianNaiveBayes_k_fold(documento,1,2) #TfidfVectorizer Bigram
#GaussianNaiveBayes_k_fold(documento,1,3) #TfidfVectorizer Trigram

def logistic_reg(documento,tipo, ngram, test_size, graba):
    dataframe = pd.read_csv(documento,sep=',')
    #dataframe.loc[dataframe.TIPO != 'F','TIPO']='NF'
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    X1 = dataframe['REQUISITO'].to_numpy()
    #y = dataframe['TIPO'].to_numpy()
    y = dataframe['ETIQUETAFINAL'].to_numpy()
    
    if tipo == 0:
        (X,count_vectorizer) = bag_of_words_CountVec(X1,ngram,0)
    if tipo == 1:
        (X,tfidf_vectorizer) = tf_idf_Vec(X1,ngram,0) 
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=0)
    print(len(X_train))
    print(len(X_test))
    print(y_test)
    logisticRegr = LogisticRegression(solver='lbfgs')
    classifier = logisticRegr.fit(X_train, y_train)
    
    y_pred = logisticRegr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred, output_dict=True)
    print('F precision:', round(cr['F']['precision'], 2))
    print('F recall:', round(cr['F']['recall'], 2))
    print('F F-measure:', round(cr['F']['f1-score'], 2))
    print('NF precision:', round(cr['NF']['precision'], 2))
    print('NF recall:', round(cr['NF']['recall'], 2))
    print('NF F-measure:', round(cr['NF']['f1-score'], 2))
    
    if graba == 1:
        if tipo == 0:
            nombre = 'logistic_BOW_'+str(ngram)    
            output = open(nombre+'_count_vectorizer.pkl',  'wb')
            dump(count_vectorizer,  output, -1)
            output.close()
        if tipo == 1:
            nombre = 'logistic_TFIDF_'+str(ngram) 
            output = open(nombre+'_tfidf_vectorizer.pkl',  'wb')
            dump(tfidf_vectorizer,  output, -1)
            output.close()
        
        output2 = open(nombre+'.pkl',  'wb')
        dump(classifier,  output2, -1)
        output2.close()

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#logistic_reg(documento,0,0,0.20,0) #BOW CountVectorizer
#logistic_reg(documento,0,2,0.20,0) #BOW CountVectorizer Bigram
#logistic_reg(documento,0,3,0.20,0) #BOW CountVectorizer Trigram
#logistic_reg(documento,1,0,0.20,0) #TfidfVectorizer
#logistic_reg(documento,1,2,0.20,0) #TfidfVectorizer Bigram
#logistic_reg(documento,1,3,0.20,0) #TfidfVectorizer Trigram

def Logictic_reg_k_fold(documento,tipo, ngram):
    accuracy_list=[]
    f_precision=[]
    f_recall=[]
    f_f_score=[]
    nf_precision=[]
    nf_recall=[]
    nf_f_score=[]
    dataframe = pd.read_csv(documento,sep=',')
    #dataframe.loc[dataframe.TIPO != 'F','TIPO']='NF'
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    dataframe2 = dataframe.sample(frac=1)    
    X1 = dataframe2['REQUISITO'].to_numpy()
    #y = dataframe2['TIPO'].to_numpy()
    y = dataframe2['ETIQUETAFINAL'].to_numpy()
    
    if tipo == 0:
        (X,count_vectorizer) = bag_of_words_CountVec(X1,ngram,0)
    if tipo == 1:
        (X,tfidf_vectorizer) = tf_idf_Vec(X1,ngram,0) 
        
    n_folds = 10
    print(len(X1))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]        
        print(len(X_train),len(X_test))
        
        logisticRegr = LogisticRegression(solver='lbfgs')
        logisticRegr.fit(X_train, y_train)
        y_pred = logisticRegr.predict(X_test)
                
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        accuracy_list.append(accuracy)
        cr = classification_report(y_test, y_pred, output_dict=True)    
        f_precision.append(cr['F']['precision'])
        f_recall.append(cr['F']['recall'])
        f_f_score.append(cr['F']['f1-score'])
        nf_precision.append(cr['NF']['precision'])
        nf_recall.append(cr['NF']['recall'])
        nf_f_score.append(cr['NF']['f1-score'])
    
    print('accuracy average', round(mean(accuracy_list), 2))
    print(accuracy_list)
    print('F precision:', round(mean(f_precision), 2))
    print('F recall:', round(mean(f_recall), 2))
    print('F F-measure:', round(mean(f_f_score), 2))
    print('NF precision:', round(mean(nf_precision), 2))
    print('NF recall:', round(mean(nf_recall), 2))
    print('NF F-measure:', round(mean(nf_f_score), 2))

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#Logictic_reg_k_fold(documento,0,0) #BOW CountVectorizer
#Logictic_reg_k_fold(documento,0,2) #BOW CountVectorizer Bigram
#Logictic_reg_k_fold(documento,0,3) #BOW CountVectorizer Trigram
#Logictic_reg_k_fold(documento,1,0) #TfidfVectorizer
#Logictic_reg_k_fold(documento,1,2) #TfidfVectorizer Bigram
#Logictic_reg_k_fold(documento,1,3) #TfidfVectorizer Trigram
    
def support_vector_machine(documento,tipo, ngram, test_size, graba):
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    X1 = dataframe['REQUISITO'].to_numpy()
    y = dataframe['ETIQUETAFINAL'].to_numpy()
    
    if tipo == 0:
        (X,count_vectorizer) = bag_of_words_CountVec(X1,ngram,0)
    if tipo == 1:
        (X,tfidf_vectorizer) = tf_idf_Vec(X1,ngram,0) 
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=0)
    print(len(X_train))
    print(len(X_test))
    print(y_test)
    
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cr = classification_report(y_test, y_pred, output_dict=True)
    #print(cr)
    print('F precision:', round(cr['F']['precision'], 2))
    print('F recall:', round(cr['F']['recall'], 2))
    print('F F-measure:', round(cr['F']['f1-score'], 2))
    print('NF precision:', round(cr['NF']['precision'], 2))
    print('NF recall:', round(cr['NF']['recall'], 2))
    print('NF F-measure:', round(cr['NF']['f1-score'], 2))
    
    if graba == 1:
        if tipo == 0:
            nombre = 'SVM_BOW_'+str(ngram)    
            output = open(nombre+'_count_vectorizer.pkl',  'wb')
            dump(count_vectorizer,  output, -1)
            output.close()
        if tipo == 1:
            nombre = 'SVM_TF_IDF_'+str(ngram)    
            output = open(nombre+'_vectorizer.pkl',  'wb')
            dump(tfidf_vectorizer,  output, -1)
            output.close()
            
        output2 = open(nombre+'.pkl',  'wb')
        dump(classifier,  output2, -1)
        output2.close()
            
#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#support_vector_machine(documento,0,0,0.20,0) #BOW CountVectorizer
#support_vector_machine(documento,0,2,0.20,0) #BOW CountVectorizer Bigram
#support_vector_machine(documento,0,3,0.20,0) #BOW CountVectorizer Trigram
#support_vector_machine(documento,1,0,0.20,0) #TfidfVectorizer
#support_vector_machine(documento,1,2,0.20,0) #TfidfVectorizer Bigram
#support_vector_machine(documento,1,3,0.20,0) #TfidfVectorizer Trigram

def support_vector_machine_k_fold(documento,tipo, ngram):
    accuracy_list=[]
    f_precision=[]
    f_recall=[]
    f_f_score=[]
    nf_precision=[]
    nf_recall=[]
    nf_f_score=[]
    dataframe = pd.read_csv(documento,sep=',')
    dataframe.loc[dataframe.ETIQUETAFINAL != 'F','ETIQUETAFINAL']='NF'
    dataframe2 = dataframe.sample(frac=1)    
    X1 = dataframe2['REQUISITO'].to_numpy()
    y = dataframe2['ETIQUETAFINAL'].to_numpy()
    
    if tipo == 0:
        (X,count_vectorizer) = bag_of_words_CountVec(X1,ngram,0)
    if tipo == 1:
        (X,tfidf_vectorizer) = tf_idf_Vec(X1,ngram,0) 
        
    n_folds = 10
    print(len(X1))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]        
        print(len(X_train),len(X_test))
        
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)    
        y_pred = clf.predict(X_test)
                
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        accuracy_list.append(accuracy)
        cr = classification_report(y_test, y_pred, output_dict=True)    
        f_precision.append(cr['F']['precision'])
        f_recall.append(cr['F']['recall'])
        f_f_score.append(cr['F']['f1-score'])
        nf_precision.append(cr['NF']['precision'])
        nf_recall.append(cr['NF']['recall'])
        nf_f_score.append(cr['NF']['f1-score'])
    
    print('accuracy average', round(mean(accuracy_list), 2))
    print(accuracy_list)
    print('F precision:', round(mean(f_precision), 2))
    print('F recall:', round(mean(f_recall), 2))
    print('F F-measure:', round(mean(f_f_score), 2))
    print('NF precision:', round(mean(nf_precision), 2))
    print('NF recall:', round(mean(nf_recall), 2))
    print('NF F-measure:', round(mean(nf_f_score), 2))

#documento = r"datos_formal_promise.csv"
#documento = r"datos_formal_pruebas_grupo2.csv"
documento = r"datos_formal_pruebas_grupo3.csv"
#support_vector_machine_k_fold(documento,0,0) #BOW CountVectorizer
#support_vector_machine_k_fold(documento,0,2) #BOW CountVectorizer Bigram
#support_vector_machine_k_fold(documento,0,3) #BOW CountVectorizer Trigram
#support_vector_machine_k_fold(documento,1,0) #TfidfVectorizer
#support_vector_machine_k_fold(documento,1,2) #TfidfVectorizer Bigram
#support_vector_machine_k_fold(documento,1,3) #TfidfVectorizer Trigram
