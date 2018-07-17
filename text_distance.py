# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 20:07:13 2018

@author: Ernest_Mkrtchyan
"""
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import dendrogram, linkage

import pandas as pd
# a custom function that just computes Euclidean distance

def all_words(p):
    allwords = set()
    for sentence in p:
        p1_words = split_sentence(sentence)
        for word in p1_words:
           allwords.add(word)
    return allwords


def split_sentence(p):
    p = p.replace("(", ' ').replace(")", ' ').replace(",", ' ').replace(".", ' ')
    return p.split(' ')


def sentence_dist(p1, p2):
    p1_map = dict(Counter(split_sentence(p1)))
    p2_map = dict(Counter(split_sentence(p2)))
    for key in allwords:
        if(p1_map.get(key) == None):
            p1_map.update({key: 0})
        if(p2_map.get(key) == None):
            p2_map.update({key: 0})
    vdot_p1 = np.vdot(list(p1_map.values()), list(p1_map.values()))
    vdot_p2 = np.vdot(list(p2_map.values()), list(p2_map.values()))
    vdot_p1p2 = np.vdot(list(p1_map.values()), list(p2_map.values()))
    dist = vdot_p1p2/((vdot_p1*vdot_p2)**0.5)
    result = np.arccos(dist)
    #print(result)
    return result

     
def metric_dist(n1, n2):
    return sentence_dist(abstract[int(n1[0])], abstract[int(n2[0])])

full_data = pd.read_csv('accepted papers/[UCI] AAAI-14 Accepted Papers - Papers.csv')        

data = full_data[['index']].values
abstract = full_data['abstract'].values

''' dendogram '''
#dend = dendrogram(linkage(data, method = 'ward')) 
dend = dendrogram(linkage(data, metric=metric_dist)) 

#clustering = ag_clustering(affinity=metric_dist, linkage='complete')

allwords = all_words(abstract)
fclust1 = fclusterdata(data, 0.8, criterion='distance', metric = metric_dist)
#fclust2 = fclusterdata(X, 50, criterion='distance', metric='euclidean')

print(fclust1)

'''        
sentence1 = 'We propose a voted dual averaging method for online classification problems with classification regularization.'
sentence2 = 'You do a voted dual averaging method for method classification problems with explicit regularization.'

 

sentence_dist(sentence1, sentence2, all_words(sentence1, sentence2))        
        '''
