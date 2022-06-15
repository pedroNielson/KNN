#!/usr/bin/env python
# coding: utf-8

# # Comparing KNN algorithms
# 
# ####  - KNN example using sklearn
# 
# First all the imports 
# 

# In[1]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import mode
from numpy.random import randint


# Loading the dataset and setting the data and target into X and y variables

# In[2]:


iris = load_iris()
iris


# In[3]:


X = iris.data
y = iris.target


# Inserting training and testing variables with the 'train_test_split' method

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Creating the Knn classifier, setting 3 neighbors and running the fit and test methods

# In[5]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# Printing the accuracy of the classifier

# In[6]:


from sklearn import metrics
print('Accuracy knn from sklearn: ', metrics.accuracy_score(y_test, y_pred))


# ### Now, building KNN from scratch
# 
# First is created a method for the euclidian distance calculation

# In[7]:


def euclidian(p1, p2):
    distance = np.sqrt(np.sum((p1-p2)**2))
    return distance


# Then the knn itself

# In[8]:


def knn_manual(k, y, input, treino):
    result = []

    for item in input:
        dist_vect = []

        for i in range(len(treino)):
            distance = euclidian(np.array(treino[i,:]),item)
            dist_vect.append(distance)
        
        dist_vect = np.array(dist_vect)


        aux = np.argsort(dist_vect)[:k]
        labels = y[aux]

        voting = mode(labels)
        voting = voting.mode[0]
        result.append(voting)
    
    return result


# Creating variables for training and testing

# In[9]:



training = xxx = randint(0, 150, 100)
X_treino = X[training]
y_treino = y[training]


# In[10]:


testing = xxx = randint(0, 150, 50)
X_teste = X[testing]
y_teste = y[testing]


# Runining the method

# In[11]:


predict = knn_manual(3, y_treino, X_teste, X_treino)


# Printing the knn from scratch accuracy

# In[12]:


print('Accuracy knn from scratch: ', metrics.accuracy_score(y_teste, predict))

