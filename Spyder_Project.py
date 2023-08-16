# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:14:35 2023

@author: mrkov
"""

import tarfile, os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import time
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Sequential
import pickle
import sklearn

#%%

tfile = tarfile.open(name='aclImdb_v1.tar.gz', mode='r:gz')
# print(tfile.getmembers())
tfile.close()

#%%
SEED = 321
PATH = './aclImdb'
tf.random.set_seed(SEED)
STOP_SYMBOLS = ['<br />'] + list('!@#$%^&*()"№;:?<>,.-_=+~|[]{}')+ ["'"]


def get_files(path, train = True, numeric = True):
    polarity = ('/pos', '/neg')
    df = pd.DataFrame(columns = ['id', 'text', 'polarity', 'rating'])
    path += '/train' if train else '/test'
    polar = (1, 0) if numeric else ('pos', 'neg')
    i = 0
    for pol in polarity:
        for file in tqdm(os.listdir(path + pol)):
            id_, rating = re.split('_', file[:-4]) 
            with open(path + pol + '/' + file, 'r', encoding='utf-8') as f:
                text = f.read()
            df.loc[0 if pd.isna(df.index.max()) else df.index.max() + 1] = [int(id_), text, polar[i], int(rating)]
        i += 1
    return df

def text_to_token(df, stop_symbols = None, stop_words = None, use_clear_text = False):
    nlp = spacy.load("en_core_web_sm")

    all_text_tokens = []
    clear_text_list = []
    for i in tqdm(range(len(df))):
        clear_text = df.iloc[i].text.lower() if not use_clear_text else df.iloc[i].clear_text
        if(stop_symbols != None):
            for s in stop_symbols:
                clear_text = clear_text.replace(s, ' ')
        doc = nlp(clear_text)   
        t = []
        for token in doc:
            if((token.text.isdigit() or token.text.isalpha()) and (stop_words == None or str(token) not in stop_words)):
                t.append(str(token))
        clear_text_list.append(' '.join(t))
        all_text_tokens.append(t)
    df['tokens'] = all_text_tokens
    df['clear_text'] = clear_text_list
    
    return df

def term_freq_coef(all_words, df):
    polarity = ('neg', 'pos')
    df_coef = pd.DataFrame(pd.Series(all_words), columns = ['word'])
    for p in tqdm(df.polarity.unique()):
        l = len(df[df.polarity == p])
        all_t = [tt for t in df[df.polarity == p].tokens for tt in t]
        cnter = Counter(all_t)
        df_coef[polarity[p]] = [cnter[w] for w in all_words]
        df_coef[str(polarity[p]) + '_coef'] = [cnter[w] / l for w in all_words]
    for r in tqdm(df.rating.unique()):
        l = len(df[df.rating == r])
        all_t = [tt for t in df[df.rating == r].tokens for tt in t]
        cnter = Counter(all_t)
        df_coef[r] = [cnter[w] for w in all_words]
        df_coef[str(r) + '_coef'] = [cnter[w] / l for w in all_words]
    coef_list = []
    for i in tqdm(range(len(df_coef))):
        coef_list.append(min(df_coef[['pos', 'neg']].iloc[i]) / max(df_coef[['pos', 'neg']].iloc[i]))
    df_coef['ratio'] = coef_list
    return df_coef

def counter_words(data):
    all_tokens = []
    for tt in tqdm(data.tokens):
        all_tokens += tt
    return Counter(all_tokens)

def word_to_vec(data, lenVec, vectorizer):
    list_wv = []
    for i in tqdm(range(len(data))):
        j = 0
        list_vect = []
        for w in data.iloc[i].tokens:
            try:
                list_vect.append(vectorizer.wv[w])
            except:
                list_vect.append(np.zeros(lenVec, dtype='f'))
            j+=1
        if (j==0):
            list_vect.append(np.zeros(lenVec, dtype='f'))
            j = 1
        
        list_wv.append(sum(list_vect) / j)
    return list_wv

def create_model_polarity():
    model1 = Sequential()
    model1.add(Embedding(15000, 300, input_length=200))
    model1.add(LSTM(150,return_sequences=False))
    model1.add(Dense(64, activation='sigmoid'))
    model1.add(Dropout(0.2))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.summary()
    return model1

def create_model_rating():
    model1 = Sequential()
    model1.add(Embedding(15000, 300, input_length=200))
    model1.add(LSTM(150,return_sequences=False))
    model1.add(Dense(64, activation='sigmoid'))
    model1.add(Dropout(0.2))
    model1.add(Dense(4, activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.summary()
    return model1



#%%
start_time = time.time()

df = get_files(PATH)
sns.histplot(data=df, x="rating", hue="polarity", legend = False, discrete = True)
plt.xticks(np.arange(10) + 1)
plt.show()
    
#%% Определяем наименее часто встречаемые слова и удаляем их

df = text_to_token(df, stop_symbols=STOP_SYMBOLS)
cnt = counter_words(df)
STOP_WORDS = set(key for key, value in cnt.items() if value < 13)
df = text_to_token(df, stop_words = STOP_WORDS, use_clear_text = True)

#%%

cnt = counter_words(df)
df_coef = term_freq_coef(cnt.keys(), df)

#%% Определяем слова, которые встречаются часто как в одном классе, так и в другом

STOP_WORDS.update(w for w in df_coef[df_coef.ratio > 0.96].word)
df = text_to_token(df, stop_words = STOP_WORDS, use_clear_text = True)
cnt = counter_words(df)
df_coef = term_freq_coef(cnt.keys(), df)

print("Этап очистки текста закончен за\n--- %s seconds ---" % (time.time() - start_time))

#%% Вектора TF-IDF
start_time = time.time()

vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(df.clear_text)

#%% Вектора Word2Vec

vectorizer_wv = Word2Vec(
    min_count=10,
    window=20,
    vector_size=300,
    negative=10,
    alpha=0.03,
    min_alpha=0.001,
    sample=6e-5,
    sg=0,
    seed = SEED)
vectorizer_wv.build_vocab(df.tokens)
vectorizer_wv.train(df.tokens, total_examples=vectorizer_wv.corpus_count, epochs=30, report_delay=1)
X_train_w2v = word_to_vec(df, vectorizer_wv.vector_size, vectorizer_wv)

print("Этап векторизации текста закончен за\n--- %s seconds ---" % (time.time() - start_time))

#%% Обучение моделей логистической регрессии
start_time = time.time()

logreg_tfidf_polarity = LogisticRegression(random_state=SEED).fit(X_train_tfidf, df.polarity)
logreg_w2v_polarity = LogisticRegression(random_state=SEED).fit(X_train_w2v, df.polarity)

#%% Обучение моделей XGB

xgb_tfidf_polarity = XGBClassifier(n_estimators = 700, learning_rate = 0.2, random_state = SEED).fit(X_train_tfidf, df.polarity)
xgb_w2v_polarity = XGBClassifier(n_estimators = 700, learning_rate = 0.2, random_state = SEED).fit(X_train_w2v, df.polarity)

print("Этап обучения моделей закончен за\n--- %s seconds ---" % (time.time() - start_time))
#%% Как показала практика, в задачах, где не требуется добиться максимальной точности, использование подбора параметров - не слишком разумно по отношению к временным ресурсам
# from sklearn.model_selection import GridSearchCV

# params = {
#         'n_estimators': [50,100,150,200,300,500],
#         'max_depth': [None, 3, 5, 7, 9]
#         }
# grid_search = GridSearchCV(XGBClassifier(), params, n_jobs=-1, cv=5)

# grid_search.fit(X_train, y_train)
#Best params max_depth : None, n_estimators : 500

#%% векторизация для NN и ее обучение
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(df, df.polarity, test_size=0.15, random_state=SEED)

tokenizer_nn = Tokenizer(num_words=15000)
tokenizer_nn.fit_on_texts(df.clear_text)
X_train_nn = tokenizer_nn.texts_to_sequences(X_train.clear_text)
X_test_nn = tokenizer_nn.texts_to_sequences(X_test.clear_text)
X_train_nn = pad_sequences(X_train_nn, maxlen=200)
X_test_nn = pad_sequences(X_test_nn, maxlen=200)
lstm = create_model_polarity()
lstm.fit(X_train_nn, y_train, batch_size=512, epochs=3, validation_data=(X_test_nn, y_test), verbose = 1)

print("Этап векторизации NN и обучения закончен за\n--- %s seconds ---" % (time.time() - start_time))
#%% Загрузка тестового набора данных и его обработка

df_test = get_files(path=PATH, train=False)
df_test = text_to_token(df_test, stop_symbols=STOP_SYMBOLS, stop_words=STOP_WORDS)

#%% Векторизация тестовой выборки
start_time = time.time()
X_test_w2v = word_to_vec(df_test, vectorizer_wv.vector_size, vectorizer_wv)
X_test_tfidf = vectorizer_tfidf.transform(df_test.clear_text)
X_test_lstm = tokenizer_nn.texts_to_sequences(df_test.clear_text)
X_test_lstm = pad_sequences(X_test_lstm, maxlen=200)

print("Этап тестовой выборки закончен за\n--- %s seconds ---" % (time.time() - start_time))
#%% Предсказания
start_time = time.time()
y_pred_logreg_tfidf_polarity = logreg_tfidf_polarity.predict(X_test_tfidf)
y_pred_logreg_w2v_polarity = logreg_w2v_polarity.predict(X_test_w2v)
y_pred_xgb_tfidf_polarity = xgb_tfidf_polarity.predict(X_test_tfidf)
y_pred_xgb_w2v_polarity = xgb_w2v_polarity.predict(X_test_w2v)
y_pred_lstm = lstm.predict(X_test_lstm)

print("Логистическая регрессия + TFIDF:\n" + metrics.classification_report(list(df_test.polarity), y_pred_logreg_tfidf_polarity, digits = 4))
print("Логистическая регрессия + Word2Vec:\n" + metrics.classification_report(list(df_test.polarity), y_pred_logreg_w2v_polarity, digits = 4))
print("XGBoost + TFIDF:\n" + metrics.classification_report(list(df_test.polarity), y_pred_xgb_tfidf_polarity, digits = 4))
print("XGBoost + Word2Vec:\n" + metrics.classification_report(list(df_test.polarity), y_pred_xgb_w2v_polarity, digits = 4))
print("LSTM:\n" + metrics.classification_report(list(df_test.polarity), [round(y) for y in y_pred_lstm.reshape(1, -1)[0]], digits = 4))

print("Этап предсказания типа обзора закончен за\n--- %s seconds ---" % (time.time() - start_time))
#%% Вектора TF-IDF (positive)

vectorizer_tfidf_pos = TfidfVectorizer()
X_train_tfidf_pos = vectorizer_tfidf_pos.fit_transform(df[df.polarity == 1].clear_text)

#%% Вектора Word2Vec (positive)

vectorizer_wv_pos = Word2Vec(
    min_count=10,
    window=20,
    vector_size=300,
    negative=10,
    alpha=0.03,
    min_alpha=0.001,
    sample=6e-5,
    sg=0,
    seed = SEED)

vectorizer_wv_pos.build_vocab(df[df.polarity == 1].tokens)
vectorizer_wv_pos.train(df[df.polarity == 1].tokens, total_examples=vectorizer_wv_pos.corpus_count, epochs=30, report_delay=1)
X_train_w2v_pos = word_to_vec(df[df.polarity == 1], vectorizer_wv_pos.vector_size, vectorizer_wv_pos)

#%% Вектора TF-IDF (negative)

vectorizer_tfidf_neg = TfidfVectorizer()
X_train_tfidf_neg = vectorizer_tfidf_neg.fit_transform(df[df.polarity == 0].clear_text)

#%% Вектора Word2Vec (negative)

vectorizer_wv_neg = Word2Vec(
    min_count=10,
    window=20,
    vector_size=300,
    negative=10,
    alpha=0.03,
    min_alpha=0.001,
    sample=6e-5,
    sg=0,
    seed = SEED)

vectorizer_wv_neg.build_vocab(df[df.polarity == 0].tokens)
vectorizer_wv_neg.train(df[df.polarity == 0].tokens, total_examples=vectorizer_wv_neg.corpus_count, epochs=30, report_delay=1)
X_train_w2v_neg = word_to_vec(df[df.polarity == 0], vectorizer_wv_neg.vector_size, vectorizer_wv_neg)

#%%
df_lstm_train = df.sample(frac = 0.85, random_state = 321)
df_lstm_test = df.drop(df_lstm_train.index)

''' Positive rating '''
logreg_tfidf_pos_r = LogisticRegression(random_state=SEED).fit(X_train_tfidf_pos, df[df.polarity == 1].rating)
logreg_w2v_pos_r = LogisticRegression(random_state=SEED).fit(X_train_w2v_pos, df[df.polarity == 1].rating)

xgb_tfidf_pos_r = XGBClassifier(n_estimators = 700, learning_rate = 0.2, random_state = SEED).fit(X_train_tfidf_pos, df[df.polarity == 1].rating)
xgb_w2v_pos_r = XGBClassifier(n_estimators = 700, learning_rate = 0.2, random_state = SEED).fit(X_train_w2v_pos, df[df.polarity == 1].rating)

tokenizer_nn_pos = Tokenizer(num_words=15000)
tokenizer_nn_pos.fit_on_texts(df[df.polarity == 1].clear_text)
X_train_nn_pos = tokenizer_nn.texts_to_sequences(df_lstm_train[df_lstm_train.polarity == 1].clear_text)
X_test_nn_pos = tokenizer_nn.texts_to_sequences(df_lstm_test[df_lstm_test.polarity == 1].clear_text)
X_train_nn_pos = pad_sequences(X_train_nn_pos, maxlen=200)
X_test_nn_pos = pad_sequences(X_test_nn_pos, maxlen=200)
lstm_pos_r = create_model_rating()
lstm_pos_r.fit(X_train_nn_pos, pd.get_dummies(df_lstm_train[df_lstm_train.polarity == 1].rating), batch_size=512, epochs=3, validation_data=(X_test_nn_pos, pd.get_dummies(df_lstm_test[df_lstm_test.polarity == 1].rating)), verbose = 1)

''' Negative rating '''
logreg_tfidf_neg_r = LogisticRegression(random_state=SEED).fit(X_train_tfidf_neg, df[df.polarity == 0].rating)
logreg_w2v_neg_r = LogisticRegression(random_state=SEED).fit(X_train_w2v_neg, df[df.polarity == 0].rating)

xgb_tfidf_neg_r = XGBClassifier(n_estimators = 700, learning_rate = 0.2, random_state = SEED).fit(X_train_tfidf_neg, df[df.polarity == 0].rating)
xgb_w2v_neg_r = XGBClassifier(n_estimators = 700, learning_rate = 0.2, random_state = SEED).fit(X_train_w2v_neg, df[df.polarity == 0].rating)

tokenizer_nn_neg = Tokenizer(num_words=15000)
tokenizer_nn_neg.fit_on_texts(df[df.polarity == 0].clear_text)
X_train_nn_neg = tokenizer_nn.texts_to_sequences(df_lstm_train[df_lstm_train.polarity == 0].clear_text)
X_test_nn_neg = tokenizer_nn.texts_to_sequences(df_lstm_test[df_lstm_test.polarity == 0].clear_text)
X_train_nn_neg = pad_sequences(X_train_nn_neg, maxlen=200)
X_test_nn_neg = pad_sequences(X_test_nn_neg, maxlen=200)
lstm_neg_r = create_model_rating()
lstm_neg_r.fit(X_train_nn_neg, pd.get_dummies(df_lstm_train[df_lstm_train.polarity == 0].rating), batch_size=512, epochs=3, validation_data=(X_test_nn_neg, pd.get_dummies(df_lstm_test[df_lstm_test.polarity == 0].rating)), verbose = 1)
#%% Векторизация тестовой выборки (positive)

X_test_w2v_pos = word_to_vec(df_test[df_test.polarity == 1], vectorizer_wv_pos.vector_size, vectorizer_wv_pos)
X_test_tfidf_pos = vectorizer_tfidf_pos.transform(df_test[df_test.polarity == 1].clear_text)
X_test_lstm_pos = tokenizer_nn_pos.texts_to_sequences(df_test[df_test.polarity == 1].clear_text)
X_test_lstm_pos = pad_sequences(X_test_lstm_pos, maxlen=200)

#%%

X_test_w2v_neg = word_to_vec(df_test[df_test.polarity == 0], vectorizer_wv_neg.vector_size, vectorizer_wv_neg)
X_test_tfidf_neg = vectorizer_tfidf_neg.transform(df_test[df_test.polarity == 0].clear_text)
X_test_lstm_neg = tokenizer_nn_neg.texts_to_sequences(df_test[df_test.polarity == 0].clear_text)
X_test_lstm_neg = pad_sequences(X_test_lstm_neg, maxlen=200)

#%%

y_pred_logreg_tfidf_pos_r = logreg_tfidf_pos_r.predict(X_test_tfidf_pos)
y_pred_logreg_w2v_pos_r = logreg_w2v_pos_r.predict(X_test_w2v_pos)
y_pred_xgb_tfidf_pos_r = xgb_tfidf_pos_r.predict(X_test_tfidf_pos)
y_pred_xgb_w2v_pos_r = xgb_w2v_pos_r.predict(X_test_w2v_pos)
y_pred_lstm_pos = lstm_pos_r.predict(X_test_lstm_pos)

print('positive')
print("Логистическая регрессия + TFIDF:\n" + metrics.classification_report(list(df_test[df_test.polarity == 1].rating), y_pred_logreg_tfidf_pos_r, digits = 4))
print("Логистическая регрессия + Word2Vec:\n" + metrics.classification_report(list(df_test[df_test.polarity == 1].rating), y_pred_logreg_w2v_pos_r, digits = 4))
print("XGBoost + TFIDF:\n" + metrics.classification_report(list(df_test[df_test.polarity == 1].rating), y_pred_xgb_tfidf_pos_r, digits = 4))
print("XGBoost + Word2Vec:\n" + metrics.classification_report(list(df_test[df_test.polarity == 1].rating), y_pred_xgb_w2v_pos_r, digits = 4))
print("LSTM:\n" + metrics.classification_report(list(df_test[df_test.polarity == 1].rating), [np.argmax(y) + 7 for y in y_pred_lstm_pos], digits = 4))

y_pred_logreg_tfidf_neg_r = logreg_tfidf_neg_r.predict(X_test_tfidf_neg)
y_pred_logreg_w2v_neg_r = logreg_w2v_neg_r.predict(X_test_w2v_neg)
y_pred_xgb_tfidf_neg_r = xgb_tfidf_neg_r.predict(X_test_tfidf_neg)
y_pred_xgb_w2v_neg_r = xgb_w2v_neg_r.predict(X_test_w2v_neg)
y_pred_lstm_neg = lstm_neg_r.predict(X_test_lstm_neg)

print('negative')
print("Логистическая регрессия + TFIDF:\n" + metrics.classification_report(list(df_test[df_test.polarity == 0].rating), y_pred_logreg_tfidf_neg_r, digits = 4))
print("Логистическая регрессия + Word2Vec:\n" + metrics.classification_report(list(df_test[df_test.polarity == 0].rating), y_pred_logreg_w2v_neg_r, digits = 4))
print("XGBoost + TFIDF:\n" + metrics.classification_report(list(df_test[df_test.polarity == 0].rating), y_pred_xgb_tfidf_neg_r, digits = 4))
print("XGBoost + Word2Vec:\n" + metrics.classification_report(list(df_test[df_test.polarity == 0].rating), y_pred_xgb_w2v_neg_r, digits = 4))
print("LSTM:\n" + metrics.classification_report(list(df_test[df_test.polarity == 0].rating), [np.argmax(y) + 1 for y in y_pred_lstm_neg], digits = 4))

#%% Save models

with open('vectorizer_tfidf_polarity.pk', 'wb') as f:
    pickle.dump(vectorizer_tfidf, f)
with open('logreg_tfidf_polarity.pk', 'wb') as f:
    pickle.dump(logreg_tfidf_polarity, f)
with open('vectorizer_tfidf_positive.pk', 'wb') as f:
    pickle.dump(vectorizer_tfidf_pos, f)
with open('xgb_tfidf_positive_r.pk', 'wb') as f:
    pickle.dump(xgb_tfidf_pos_r, f)
with open('vectorizer_wv_negative.pk', 'wb') as f:
    pickle.dump(vectorizer_wv_neg, f)
with open('logreg_w2v_neg_r.pk', 'wb') as f:
    pickle.dump(logreg_w2v_neg_r, f)
with open('stop_words.txt','wb') as f:
    pickle.dump(STOP_WORDS, f)