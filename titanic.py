# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:46:48 2018

@author: Luiza_Kharatyan
"""

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering as ag_clustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_column(data):        
    floateddata = data.astype(float)
    diff = floateddata.max() - floateddata.min()
    minEl = floateddata.min()
    data = data.astype(float).map(lambda x: (x-minEl)/diff)
    return data


# Loading the data
train = pd.read_csv('train.csv')
#train = train[train['Sex'].str.strip()!='female']
train = train.dropna(subset=['Age'])
train['Age'] = normalize_column(train['Age'])

    
train.loc[ train['Sex'] == 'male', 'Sex']  = -1
train.loc[train['Sex'] =='female' , 'Sex'] = 1
data=train[['Pclass', 'Age', 'Survived']].values

''' dendogram '''
dend = dendrogram(linkage(data, method = 'ward')) 

clustering = ag_clustering(n_clusters=2)

model = clustering.fit_predict(X=data)



plt.scatter(data[model == 0, 0], data[model == 0, 1], data[model == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data[model == 1, 0], data[model == 1, 1], data[model == 1, 2], s = 100, c = 'blue', label = 'Cluster 2')
#plt.scatter(data[model == 2, 0], data[model == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
#plt.scatter(data[model == 3, 0], data[model == 3, 1], s = 100, c = 'yellow', label = 'Cluster 4')

plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.scatter(data[model == 0, 0], data[model == 0, 1], data[model == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(data[model == 1, 0], data[model == 1, 1], data[model == 1, 2], s = 100, c = 'blue', label = 'Cluster 2')

ax.legend()

plt.show()