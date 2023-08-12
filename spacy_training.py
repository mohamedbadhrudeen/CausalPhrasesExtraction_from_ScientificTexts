# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:34:00 2023

@author: Choungryeol Axl Lee
"""

import re
import spacy as sp
import pandas as pd


# Load the spacy model
nlp_ = sp.load('en_core_web_lg')

# Data had some white spaces within the text, the function
# below removes that.
def remove_whitespace(text):
    return re.sub(r'\s+([?.,:;!"])', r'\1', text)


# Read the training, test and dev data
training_data = pd.read_csv('./data/train_data.csv')
dev_data = pd.read_csv('./data/dev_data.csv')


training_data['text'] = training_data['text'].apply(lambda x: remove_whitespace(x))
dev_data['text'] = dev_data['text'].apply(lambda x: remove_whitespace(x))


# Converting the data in csv file to spacy format 
def csv_to_spacy(data, outfile):
    db = sp.tokens.DocBin()
   
    for doc, label in nlp_.pipe(zip(data['text'], data['label']), as_tuples=True):
        doc.cats["CAUSAL"] = label == 1
        doc.cats["NOTCAUSAL"] = label == 0
        db.add(doc)
    
    db.to_disk(outfile)

csv_to_spacy(training_data, 'causal_training_data.spacy')
csv_to_spacy(dev_data, 'causal_dev_data.spacy')

