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
    p = p.replace("(", ' ').replace(")", ' ').replace(",", ' ').replace(".", ' ').lower().replace(" a ", ' ').replace(" the ", ' ')
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
    return result

     
def metric_dist(n1, n2):
    result = sentence_dist(abstract[int(n1[0])], abstract[int(n2[0])])
    dist_dict[str(n1) + str(n2)] = result
    return result

full_data = pd.read_csv('res/Papers_dev100.csv')        

data = full_data[['index']].values
abstract = full_data['abstract'].values

allwords = all_words(abstract)
dist_dict = {}

''' dendogram '''
#dend = dendrogram(linkage(data, method = 'ward')) 
dend = dendrogram(linkage(data, metric=metric_dist))

#clustering = ag_clustering(affinity=metric_dist, linkage='complete')

for i in range(3):    
    t = 0.74 + i * 0.03
    fclust1 = fclusterdata(data, t, criterion='distance', metric = metric_dist)
    
    unique, counts = np.unique(fclust1, return_counts=True)
    dict(zip(unique, counts))
    print(dict(zip(unique, counts)))

metric_dist([4], [5])

   
sentence1 = 'We propose a voted dual averaging method for online classification problems with classification regularization.'
sentence2 = 'You do a voted dual averaging method for method classification problems with explicit regularization.'
s3 = "Transfer learning considers related but distinct tasks defined on heterogenous domains and tries to transfer knowledge between these tasks to improve generalization performance. It is particularly useful when we do not have sufficient amount of labeled training data in some tasks, which may be very costly, laborious, or even infeasible to obtain. Instead, learning the tasks jointly enables us to effectively increase the amount of labeled training data. In this paper, we formulate a kernelized Bayesian transfer learning framework that is a principled combination of kernel-based dimensionality reduction models with task-specific projection matrices to find a shared subspace and a coupled classification model for all of the tasks in this subspace. Our two main contributions are: (i) two novel probabilistic models for binary and multiclass classification, and (ii) very efficient variational approximation procedures for these models. We illustrate the generalization performance of our algorithms on two different applications. In computer vision experiments, our method outperforms the state-of-the-art algorithms on nine out of 12 benchmark supervised domain adaptation experiments defined on two object recognition data sets. In cancer biology experiments, we use our algorithm to predict mutation status of important cancer genes from gene expression profiles using two distinct cancer populations, namely, patient-derived primary tumor data and in-vitro-derived cancer cell line data. We show that we can increase our generalization performance on primary tumors using cell lines as an auxiliary data source."
s4 = 'Transfer learning uses relevant auxiliary data to help the learning task in a target domain where labeled data are usually insufficient to train an accurate model. Given appropriate auxiliary data, researchers have proposed many transfer learning models. How to find such auxiliary data, however, is of little research in the past. In this paper, we focus on this auxiliary data retrieval problem, and propose a transfer learning framework that effectively selects helpful auxiliary data from an open knowledge space (e.g. the World Wide Web). Because there is no need of manually selecting auxiliary data for different target domain tasks, we call our framework Source Free Transfer Learning (SFTL). For each target domain task, SFTL framework iteratively queries for the helpful auxiliary data based on the learned model and then updates the model using the retrieved auxiliary data. We highlight the automatic constructions of queries and the robustness of the SFTL framework. Our experiments on the 20 NewsGroup dataset and the Google search snippets dataset suggest that the new framework is capable to have the comparable performance to those state-of-the-art methods with dedicated selections of auxiliary data.'
s5 = 'The probabilistic serial (PS) rule is one of the most well-established and desirable rules for the random assignment problem. We present the egalitarian simultaneous reservation (ESR) social decision scheme — an extension of PS to the more general setting of randomized social choice. ESR also generalizes an egalitarian rule from the literature which is defined only for dichotomous preferences. We consider various desirable fairness, efficiency, and strategic properties of ESR and show that it compares favourably against other social decision schemes. Finally, we define a more general class of social decision schemes called Simultaneous Reservation (SR), that contains ESR as well as the serial dictatorship rules. We show that outcomes of SR characterize efficiency with respect to a natural refinement of stochastic dominance.'

allwords = all_words(["the Big cat", "the aa dog", "as sas sas"])
sentence_dist(s4, s3)
