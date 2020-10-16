



import os,sys,time,re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))

from logs.logger import logger
from configs.config import *
from redis_cache.redis_connect import  conn_redis
from redis_cache.pg_connect import PostgreSql

import pandas as pd
import numpy as  np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import state_union,stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer


import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def  sentences(pg, redis_conn):
    conn, cur = pg.getConnect()
    sql = "select tid,title,content from funbox_topic where date(create_time)=date(now()) " \
          "and  language_type ='en'"
    rows= pg.selectAll(cur, sql)
    pg.closeConnect(conn, cur)

    df =pd.DataFrame(rows,columns=['tid','title','content'])
    df['content'] = df['content'].map( clearn)
    df['text'] = df['title']+df['content']
    df['title_ner'] = df['title'].map(nltk_ner)
    df['words'] = df['text'].map(nltk_words)
    sentences = df['words'].sum()
    # 词向量训练模型
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    df['tid_vec']=df['title_ner'].apply(cal_vect,args=(model,))

    #写入pg库

    return sentences

# 提取标题实体
def nltk_ner(text):
    sent = sent_tokenize(text)
    ner =[]
    for words in sent:
        word = word_tokenize(words)
        tagged = nltk.pos_tag(word)
        for t in tagged:
            if t[1]  in ('NNP','NNPS'):
                ner.append(t[0])
        # namedEnt = nltk.ne_chunk(tagged, binary=False)
        # for i in namedEnt:
        #     print(i)
        # print('nltk_ner : ',ner)
    return ner

# 还原词性+分词
def nltk_words(text):
    wnl = WordNetLemmatizer()
    sentences = sent_tokenize(text)
    # stopword = stopwords.words('english')
    sent = []
    for sentence in sentences:
        words_pos_tag = nltk.pos_tag(word_tokenize(sentence))
        words=[]
        for word, tag in words_pos_tag:
            if tag.startswith('NN'):
                words.append(wnl.lemmatize(word, pos='n'))
            if tag.startswith('VB'):
                words.append(wnl.lemmatize(word, pos='v'))
            if tag.startswith('JJ'):
                words.append(wnl.lemmatize(word, pos='a'))
            if tag.startswith('R'):
                words.append(wnl.lemmatize(word, pos='r'))
        sent.append(words)
    return sent

# 清楚html格式
def clearn(html):
    pat = re.compile('<[^>]+>', re.S)
    text = pat.sub('', html)
    return text

def cal_vect(ner,model):
    vec = np.zeros(100)
    for word in ner:
        tmp = model[word]
        vec += tmp
    return vec



if __name__ == '__main__':
    pg = PostgreSql()
    redis_conn = conn_redis()
    sentences = sentences(pg, redis_conn)



