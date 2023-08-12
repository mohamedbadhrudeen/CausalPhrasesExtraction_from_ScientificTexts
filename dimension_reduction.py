# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:42:59 2023

@author: Choungryeol Axl Lee
"""

import spacy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap


# Read the saved embeddings data from the CausalPhrasesExtraction.py
embed_ = pd.read_csv('causal_phrases_embeddings.csv')

# Principal Component Analysis
pca = PCA(n_components = 2) # 2 dimensional PCA
pca.fit(embed_)
df_pca = pca.transform(embed_)

plt.scatter(df_pca[:,0], df_pca[:,1]) # plot the two components

# Isomap dimension reduction
iso = Isomap(n_components = 2)
df_iso = iso.fit_transform(embed_) # 2 dimensional Isomap

plt.scatter(df_iso[:,0], df_iso[:,1]) # plot the two components

# TSNE dimension reduction
tsne = TSNE(n_components = 2) # 2 dimensional PCA
df_tsne = tsne.fit_transform(embed_)

plt.scatter(df_tsne[:,0], df_tsne[:,1]) # plot the two components


