# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:53:04 2018

@author: Luiza_Kharatyan
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import r2_score
from sklearn.cluster import AgglomerativeClustering as ag_clustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

europe = ('Austria', 'Belgium', 'Amsterdam', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany', 'Hungary', 'Iceland', 'Ireland', 
          'Italy', 'Lithuania', 'Netherlands', 'Poland', 'Portugal', 'Scotland', 'Spain', 'Sweden', 'Switzerland', 'U.K.', 'Wales')
america = ('Haiti','Chile', 'Argentina', 'Bolivia', 'Belize', 'Brazil', 'Canada', 'Colombia', 'Costa Rica', 'Carribean', 'Domincan Republic', 'Dominican Republic', 'Ecuador', 'Eucador', 
           'Grenada', 'Guatemala', 'Honduras', 'Mexico', 'Martinique', 'Nicaragua', 'Peru', 'Philippines', 'Puerto Rico', 'Saint Lucia', 
           'Suriname', 'U.S.A.', 'Venezuela', 'Central and S. America', 'Colombia, Ecuador', 'Costa Rica, Ven', 'Cuba', 'Ecuador, Costa Rica', 'El Salvador'
           'Haiti ', 'Hawaii', 'Jamaica', 'Panama', 'South America', 'Peru, Dom. Rep', 'Peru, Ecuador', 'DR, Ecuador, Peru', 'Peru, Belize', 'Peru, Ecuador, Venezuela',
           'St Lucia', 'St. Lucia', 'Tobago', 'Trinidad', 'Trinidad, Tobago', 'Trinidad-Tobago', 'Trinidad, Ecuador', 'Ven, Bolivia, D.R.', 'Ven, Trinidad, Ecuador', 
           'Ven.,Ecu.,Peru,Nic.', 'Venezuela, Trinidad', 'Venezuela, Carribean', 'Venezuela, Dom. Rep.', 'Trinitario')
africa = ('Cameroon', 'Congo', 'Ghana', 'Madagascar', 'Sao Tome', 'South Africa', 'Suriname', 'Kongo', 'Gabon', 'Ghana & Madagascar', 'Ivory Coast', 'Liberia', 'Nigeria',
          'Principe', 'Sao Tome & Principe', 'Tanzania', 'Uganda', 'West Africa')
asia = ('Austria', 'India', 'Israel', 'Japan', 'Russia', 'Singapore', 'South Korea', 'Vietnam', 'Burma', 'Malaysia', 'Sri Lanka', 'Togo', 'Vietnam')
australia = ('Australia', 'Fiji', 'New Zeland', 'Indonesia', 'Samoa', 'Papua New Guinea', 'Solomon Island', 'Vanuatu')
other = ('A', 'Dom. Rep., Madagascar', 'Dominican Rep., Bali', 'Ecuador, Mad., PNG' 
         'Ghana, Domin. Rep', 'Ghana, Panama, Ecuador', 'Gre., PNG, Haw., Haiti, Mad', 'Guat., D.R., Peru, Mad., PNG', 'Indonesia, Ghana', 'Madagascar, Java, PNG',
         'Madagascar & Ecuador', 'Peru, Mad., Dom. Rep.', 'Peru, Madagascar', 'Peru(SMartin,Pangoa,nacional)', 'PNG, Vanuatu, Mad', 'South America, Africa',
         'Venez,Africa,Brasil,Peru,Mex', 'Ven., Indonesia, Ecuad.', 'Ven., Trinidad, Mad.', 'Carribean(DR/Jam/Tri)', 'Venezuela, Ghana',
         'Venez,Africa,Brasil,Peru,Mex', 'Venezuela, Java', 'Venezuela/ Ghana', 'Nacional (Arriba)',  'Nacional', 'Ecuador, Mad., PNG', 'Mad., Java, PNG')

country = europe + america + africa + asia + australia + other

bean_blended = ('Blend', 'Amazon, ICS', 'Blend-Forastero,Criollo', 'Criollo, Forastero', 'Criollo, Trinitario', 'Forastero, Trinitario',
                'Trinitario, Criollo', 'Trinitario, Forastero', 'Trinitario, Nacional', 'Trinitario, TCGA')


def normalize_column(data):        
    floateddata = data.astype(float)
    diff = floateddata.max() - floateddata.min()
    minEl = floateddata.min()
    data = data.astype(float).map(lambda x: (x-minEl)/diff)
    return data

def binarize_column(data): 
    return label_binarize(data.as_matrix(columns=None), classes=data.unique())


def preprocess_dataset(dataset): 
    dataset.loc[dataset['Broad Bean Origin'].isin(europe), 'Broad Bean Origin'] = 1
    dataset.loc[dataset['Broad Bean Origin'].isin(america), 'Broad Bean Origin'] = 4
    dataset.loc[dataset['Broad Bean Origin'].isin(africa), 'Broad Bean Origin'] = 5
    dataset.loc[dataset['Broad Bean Origin'].isin(asia),  'Broad Bean Origin'] = 6
    dataset.loc[dataset['Broad Bean Origin'].isin(australia), 'Broad Bean Origin'] = 3
    dataset.loc[dataset['Broad Bean Origin'].isin(other), 'Broad Bean Origin'] = 2
    
    return dataset
    
    
full_data = pd.read_csv('chocolate-bar-ratings/flavors_of_cacao_.csv')
full_data = full_data[full_data['Bean Type'].str.strip()!='A']

full_data =  preprocess_dataset(full_data)

data=full_data[['Broad Bean Origin', 'Rating']].values

''' dendogram '''
dend = dendrogram(linkage(data, method = 'ward')) 

clustering = ag_clustering(n_clusters=6)

model = clustering.fit_predict(X=data)



plt.scatter(data[model == 0, 0], data[model == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data[model == 1, 0], data[model == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data[model == 2, 0], data[model == 2, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(data[model == 3, 0], data[model == 3, 1], s = 100, c = 'gray', label = 'Cluster 2')
plt.scatter(data[model == 4, 0], data[model == 4, 1], s = 100, c = 'green', label = 'Cluster 1')
plt.scatter(data[model == 5, 0], data[model == 5, 1], s = 100, c = 'cyan', label = 'Cluster 1')