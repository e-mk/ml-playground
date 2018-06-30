# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:27:30 2018

@author: Luiza_Kharatyan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsRegressor

europe = ('Austria', 'Belgium', 'Amsterdam', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany', 'Hungary', 'Iceland', 'Ireland', 
          'Italy', 'Lithuania', 'Netherlands', 'Poland', 'Portugal', 'Scotland', 'Spain', 'Sweden', 'Switzerland', 'U.K.', 'Wales')
america = ('Chile', 'Argentina', 'Bolivia', 'Belize', 'Brazil', 'Canada', 'Colombia', 'Costa Rica', 'Carribean', 'Dominican Republic', 'Ecuador', 'Eucador', 
           'Grenada', 'Guatemala', 'Honduras', 'Mexico', 'Martinique', 'Nicaragua', 'Peru', 'Philippines', 'Puerto Rico', 'Saint Lucia', 
           'Suriname', 'U.S.A.', 'Venezuela', 'Central and S. America', 'Colombia, Ecuador', 'Costa Rica, Ven', 'Cuba', 'Ecuador, Costa Rica', 'El Salvador'
           'Haiti', 'Hawaii', 'Jamaica', 'Panama', 'South America', 'Peru, Dom. Rep', 'Peru, Ecuador', 'DR, Ecuador, Peru', 'Peru, Belize', 'Peru, Ecuador, Venezuela',
           'St Lucia', 'Tobago', 'Trinidad', 'Trinidad, Tobago', 'Trinidad-Tobago', 'Trinidad, Ecuador', 'Ven, Bolivia, D.R.', 'Ven, Trinidad, Ecuador', 
           'Ven.,Ecu.,Peru,Nic.', 'Venezuela, Trinidad', 'Venezuela, Carribean', 'Venezuela, Dom. Rep.', 'Trinitario')
africa = ('Cameroon', 'Ghana', 'Madagascar', 'Sao Tome', 'South Africa', 'Suriname', 'Kongo', 'Gabon', 'Ghana & Madagascar', 'Ivory Coast', 'Liberia', 'Nigeria',
          'Principe', 'Sao Tome & Principe', 'Tanzania', 'Uganda', 'West Africa')
asia = ('Austria', 'India', 'Israel', 'Japan', 'Russia', 'Singapore', 'South Korea', 'Vietnam', 'Burma', 'Malaysia', 'Sri Lanka', 'Togo', 'Vietnam')
australia = ('Australia', 'Fiji', 'New Zeland', 'Indonesia', 'Samoa', 'Papua New Guinea', 'Solomon Island', 'Vanuatu')
other = ('A', 'Dom. Rep., Madagascar', 'Dominican Rep., Bali', 'Ecuador, Mad., PNG' 
         'Ghana, Domin. Rep', 'Ghana, Panama, Ecuador', 'Gre., PNG, Haw., Haiti, Mad', 'Guat., D.R., Peru, Mad., PNG', 'Indonesia, Ghana', 'Madagascar, Java, PNG',
         'Madagascar & Ecuador', 'Peru, Mad., Dom. Rep.', 'Peru, Madagascar', 'Peru(SMartin,Pangoa,nacional)', 'PNG, Vanuatu, Mad', 'South America, Africa',
         'Venez,Africa,Brasil,Peru,Mex', 'Ven., Indonesia, Ecuad.', 'Ven., Trinidad, Mad.', 
         'Venez,Africa,Brasil,Peru,Mex', 'Venezuela, Java', 'Venezuela/ Ghana', 'Nacional (Arriba)',  'Nacional')

country = europe + america + africa + asia + australia + other

def normalize_column(data):        
    floateddata = data.astype(float)
    diff = floateddata.max() - floateddata.min()
    minEl = floateddata.min()
    data = data.astype(float).map(lambda x: (x-minEl)/diff)
    return data

def binarize_column(data): 
    return label_binarize(data.as_matrix(columns=None), classes=data.unique())

def preprocess_dataset(dataset): 
    Y = normalize_column(dataset['Rating'])
    
    dataset = dataset.drop( ['Specific Bean Origin or Bar Name', 'Rating', 'Company'], axis = 1)
    dataset['Cocoa Percent'] = dataset['Cocoa Percent'].map(lambda x: str(x)[:-1])
    
    dataset['Cocoa Percent'] = normalize_column(dataset['Cocoa Percent'])
    dataset['REF'] = normalize_column(dataset['REF'])
    dataset['Review Date'] = normalize_column(dataset['Review Date'])
    
    
    dataset.loc[dataset['Company Location'].isin(europe), 'Company Location'] = 'Europe'
    dataset.loc[dataset['Company Location'].isin(america), 'Company Location'] = 'America'
    dataset.loc[dataset['Company Location'].isin(africa), 'Company Location'] = 'Africa'
    dataset.loc[dataset['Company Location'].isin(asia), 'Company Location'] = 'Asia'
    dataset.loc[dataset['Company Location'].isin(australia), 'Company Location'] = 'Australia'
    dataset.loc[dataset['Company Location'].isin(other), 'Company Location'] = 'Other'
    
    dataset.loc[dataset['Broad Bean Origin'].isin(europe), 'Broad Bean Origin'] = 'Europe'
    dataset.loc[dataset['Broad Bean Origin'].isin(america), 'Broad Bean Origin'] = 'America'
    dataset.loc[dataset['Broad Bean Origin'].isin(africa), 'Broad Bean Origin'] = 'Africa'
    dataset.loc[dataset['Broad Bean Origin'].isin(asia),  'Broad Bean Origin'] = 'Asia'
    dataset.loc[dataset['Broad Bean Origin'].isin(australia), 'Broad Bean Origin'] = 'Australia'
    dataset.loc[dataset['Broad Bean Origin'].isin(other), 'Broad Bean Origin'] = 'Other'
    
   
    dataset=pd.get_dummies(dataset, columns=['Company Location'])
    dataset=pd.get_dummies(dataset, columns=['Broad Bean Origin'])    
    dataset=pd.get_dummies(dataset, columns=['Bean Type']) 
    
    return dataset, Y
    

# Loading the data
full_data = pd.read_csv('chocolate-bar-ratings/flavors_of_cacao_.csv')
full_data, Y = preprocess_dataset(full_data)
trainset, testset = train_test_split(full_data, test_size=0.2)
Y_train, Y_test = train_test_split(Y.to_frame(name=None), test_size=0.2)
#type (full_data['Company Location'][0])
#type(trainset)
trainset.columns
neigh = KNeighborsRegressor(n_neighbors=5)

neigh.fit(trainset, Y_train) 
neigh.predict(testset)


predicted = neigh.predict(testset)
type(predicted)
actual = Y_test.as_matrix(columns=None)
(actual - predicted).mean()