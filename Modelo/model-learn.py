#https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

df_clean = pd.read_csv(r'D:\Investigacion Word2Vec\Modelo\pubmed_sentences.csv')

#Bigrams

from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in df_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram = Phraser(phrases)
sentences = bigram[sent]

#Most Frequent Words:
word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

#Training the model
import multiprocessing

from gensim.models import Word2Vec

cores = multiprocessing.cpu_count() # Count the number of cores in a computer

w2v_model = Word2Vec(min_count=20,
                     window=8,
                     size=200,
                     sample=6e-5, 
                     alpha=0.01, 
                     min_alpha=0.0001, 
                     negative=20,
                     workers=cores-1,
                     sg=1)

#Building the Vocabulary Table
t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#Training of the model:
t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

#Save the Model
w2v_model.save("w2v_model")
model = Word2Vec.load("w2v_model")

#Exploring the model
#print(w2v_model.wv.most_similar(positive=["homer"]))


