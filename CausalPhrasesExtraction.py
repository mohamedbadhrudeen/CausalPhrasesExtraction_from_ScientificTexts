# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:27:24 2023

@author: Choungryeol Axl Lee
"""

import os
import re
import glob
import numpy as np
import spacy as sp
import pandas as pd
import spacy_transformers
from spacy import displacy
from operator import concat
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sentence_transformers import SentenceTransformer, util


# Texts from Transport Policy journal, abstracts only.
abstract = []
keywords = []
year = []
# read in the whole file
Corpuses = glob.glob("./articles/*.txt")

for file in Corpuses:
    with open(file, "r", encoding="utf-8") as f:
        inputdata = f.readlines()
    
    for i in inputdata:
        if 'Abstract:' in i:
            temp_text = i.split(':', 1)[1].strip()
            if 'ABSTRACT' not in temp_text:
                abstract.append(temp_text)
        if 'Keywords:' in i:
            keywords.append(i.split(':')[1].strip())
    
    d = [i-1 for i, j in enumerate(inputdata) if 'Pages' in j]
    Year = [inputdata[i].strip()[:-1] for i in d]
    year.append(Year)
    
   
year = reduce(concat, year)
data = pd.DataFrame(list(zip(abstract, keywords, year)), columns= ['Abstract', 'Keywords', 'Year'])

# Load the trained best model, and load the default model 
# for sentence separation

nlp = sp.load('model-best')
nlp_ = sp.load('en_core_web_lg')

# Converting abstract to individual sentences using spacy default model
i = -1    
Sentence = []
abs_id = []

for abstract in data.Abstract:
    i = i+1
    s_ = nlp_(abstract)
    for sent in s_.sents:
        if len(sent) > 2:
            Sentence.append(sent.text)
            abs_id.append(i)

df = pd.DataFrame([Sentence, abs_id]).transpose()
df.columns = ['sentence', 'abstractID']


# Causal Phrases extraction from the transport policy literature
causal_sentence = []

for index, row in df.iterrows():
    d = nlp(row['sentence'])
    if d.cats['CAUSAL'] > 0.9:
        causal_sentence.append([row['sentence'], row['abstractID']])
        
# Save the file
phrases_causal = pd.DataFrame(causal_sentence, columns=['sentence', 'abstractID'])
#phrases_causal.to_csv('Causal Phrases.csv', index = False, encoding = 'utf-8')

# Import the model that was trained on scientific literature corpus
model = SentenceTransformer('allenai-specter')

# find the embeddings for each causal sentence
corpus_embeddings = model.encode(phrases_causal.sentence.to_list(), convert_to_tensor=True)
corpus_embeddings.to_csv('causal_phrases_embeddings.csv', index = False)



