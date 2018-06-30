import pandas as pd
import numpy as np
# Loading the data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

# Showing overview of the train dataset

from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset, testset = train_test_split(train, test_size=0.2)


dataset = normalizedataset(dataset)
testset = normalizedataset(testset)

clf = tree.DecisionTreeClassifier(max_depth = 4)
clf = clf.fit(dataset[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values, dataset['Survived'])
predicted = clf.predict(testset[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values)

actual = testset['Survived'].as_matrix(columns=None)

np.count_nonzero(actual - predicted)

len(actual)
import graphviz 
dot_data = tree.export_graphviz(clf, out_file = None) 
 
graph = graphviz.Source(dot_data) 
graph

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
rf = rf.fit(dataset[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values, dataset['Survived'])
predicted = rf.predict(testset[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values)

actual = testset['Survived'].as_matrix(columns=None)

np.count_nonzero(actual - predicted)

def normalizedataset(dataset):
    dataset = dataset.drop( ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
    
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 16, 'Age']  = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
    
    dataset.loc[ dataset['Sex'] == 'male', 'Sex']  = 0
    dataset.loc[dataset['Sex'] =='female' , 'Sex'] = 1
   
    dataset.loc[dataset['Fare'].isnull(), 'Fare'] = 0
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']  = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset.loc[ dataset['Embarked'] == 'S', 'Embarked']  = 0
    dataset.loc[ dataset['Embarked'] == 'C', 'Embarked']  = 1
    dataset.loc[ dataset['Embarked'] == 'Q', 'Embarked']  = 2
    
    
    dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = 0
    return dataset