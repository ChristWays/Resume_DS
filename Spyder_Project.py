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

#%%

tf = tarfile.open(name='aclImdb_v1.tar.gz', mode='r:gz')

#%%

tf.getmembers()

#%%

path = './aclImdb'

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

#%%

stop_symbols = ['<br />'] + list('!@#$%^&*()"№;:?<>,.-_=+~|[]{}')+ ["'"]


#%%

df = get_files(path)

#%%


import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=df, x="rating", hue="polarity", legend = False, discrete = True)
plt.xticks(np.arange(10) + 1)
plt.show()
    

#%%
import time
import spacy
def text_to_token(df, stop_symbols = None, stop_words = None, use_clear_text = False):
    nlp = spacy.load("en_core_web_sm")
    start_time = time.time()
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
    print("--- %s seconds ---" % (time.time() - start_time))
    return df
#%%
df = text_to_token(df, stop_symbols=stop_symbols)

#%%
all_tokens = []
for tt in tqdm(df.tokens):
    all_tokens += tt


#%%
from collections import Counter

cnt = Counter(all_tokens)
stop_words = set(key for key, value in cnt.items() if value < 13)
df = text_to_token(df, stop_words = stop_words, use_clear_text = True)


#%%

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
#%%

all_tokens = []
for tt in tqdm(df.tokens):
    all_tokens += tt
cnt = Counter(all_tokens)
df_coef = term_freq_coef(cnt.keys(), df)

#%%
stop_words.update(w for w in df_coef[df_coef.ratio > 0.96].word)
df = text_to_token(df, stop_words = stop_words, use_clear_text = True)
all_tokens = []
for tt in tqdm(df.tokens):
    all_tokens += tt
cnt = Counter(all_tokens)
df_coef = term_freq_coef(cnt.keys(), df)

#%%

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.clear_text)
#%%

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df.polarity, test_size=0.10, random_state=132)
clf1 = LogisticRegression(random_state=132).fit(X_train, y_train)

#%%

from sklearn import metrics

y_pred = clf1.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

#%%

# from sklearn.svm import SVC

# clf = SVC(random_state = 132).fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(metrics.classification_report(y_test, y_pred))


#%%

from gensim.models import Word2Vec

vectorizer_wv = Word2Vec(
    min_count=10,
    window=20,
    vector_size=300,
    negative=10,
    alpha=0.03,
    min_alpha=0.001,
    sample=6e-5,
    sg=0,
    seed = 132)
vectorizer_wv.build_vocab(df.tokens)
vectorizer_wv.train(df.tokens, total_examples=vectorizer_wv.corpus_count, epochs=30, report_delay=1)



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


#%%

list_wv = word_to_vec(df, vectorizer_wv.vector_size, vectorizer_wv)
X_train, X_test, y_train, y_test = train_test_split(list_wv, df.polarity, test_size=0.05, random_state=132)
clf = LogisticRegression(random_state=132, C=1).fit(X_train, y_train)

#%%
from sklearn import metrics
y_pred = clf.predict(X_train)
print(metrics.classification_report(y_train, y_pred))


#%%

df_test = get_files(path=path, train=False)
df_test = text_to_token(df_test, stop_symbols=stop_symbols, stop_words=stop_words)
list_wv_test = word_to_vec(df_test, vectorizer_wv.vector_size, vectorizer_wv)
y_pred = clf.predict(list_wv_test)
print(metrics.classification_report(list(df_test.polarity), y_pred))
#%%
from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# params = {
#         'n_estimators': [50,100,150,200,300,500],
#         'max_depth': [None, 3, 5, 7, 9]
#         }
# grid_search = GridSearchCV(XGBClassifier(), params, n_jobs=-1, cv=5)

# grid_search.fit(X_train, y_train)
#Best params max_depth : None, n_estimators : 500

#%%

xgb = XGBClassifier(n_estimators = 500)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(list_wv_test)
print(metrics.classification_report(list(df_test.polarity), y_pred))

#%%
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
tf.random.set_seed(321)

X_train, X_test, y_train, y_test = train_test_split(df.clear_text, df.polarity, test_size=0.10, random_state=132)
tokenizer = Tokenizer(num_words=15000)
tokenizer.fit_on_texts(df.clear_text)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(df_test.clear_text)


#%%

X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)


    
#%% Save models

from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Sequential


model1 = Sequential()
model1.add(Embedding(15000, 300, input_length=200))
model1.add(LSTM(150,return_sequences=False))
model1.add(Dense(64, activation='sigmoid'))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()

# modelXGB.save_model("modelXGB.json")

#%%

model1.fit(X_train, y_train, batch_size=512, epochs=3, validation_data=(X_test, y_test), verbose = 1)

#%%

y_pred = model1.predict(X_test)
print(metrics.classification_report(list(df_test.polarity), [round(y) for y in y_pred.reshape(1, -1)[0]]))
