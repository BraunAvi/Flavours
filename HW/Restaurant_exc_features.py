import codecs
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# print ('path: ',os.path)
os.chdir('/Users/avibraun/Dropbox/Learning/Machine Learning/IC-2017/HW1- restauraunt')
# Read training dataset:
with codecs.open('1-restaurant-train.csv') as f:
    labels, reviews = zip(*[line.split('\t') for line in f.readlines()])

print (labels[:10])
print (reviews[:2])

# Read test dataset:
with codecs.open('1-restaurant-test.csv') as f:
    kaggle_test_reviews = f.readlines()

# compute simple statistics:
def compute_data_expressions(reviews):
    features = []
    # length of each string
    features.append(map(len, reviews))
    # number of letters, digits, spaces = words
    for pattern in [str.isalpha, str.isdigit, str.isspace]:
        features.append(map(lambda review: sum(map(pattern, review)), reviews))

    features = np.array(features).T
    return features

features = compute_data_expressions(reviews)
kaggle_test_features = compute_data_expressions(kaggle_test_reviews)

print ('features:',features)
# convert labels to int values
# print ('lables before :', labels)
# lables=[int(lable) for lable in (labels[:])]
# print (type(labels))
labels = map(int, labels)
print ('lables:', labels)

# print (type(labels))

# Making problem simpler: convert to positive/negative reviews.
labels=list(labels)

answers = (np.array(labels)) >=4

print (answers, features)

from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=2)
knn_clf.fit(features, answers) # train an algorithm
print (roc_auc_score(answers, knn_clf.predict_proba(features)[:, 1]))



# Cross validation / overfitting
from sklearn.cross_validation import train_test_split
trainX, testX, trainY, testY = train_test_split(features, answers, random_state=42)


# Testing the n_neighbor parameter:


trainV = []
testV = []
n = range(2, 11, 2)

for i in n:
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(trainX, trainY)
    testV.append(roc_auc_score(testY, knn_clf.predict_proba(testX)[:, 1]))
    trainV.append(roc_auc_score(trainY, knn_clf.predict_proba(trainX)[:, 1]))
    print 'n_neighbors=', i, 'test', testV[-1], '; train', trainV[-1]

plt.plot(n,trainV, label='training set')
plt.plot(n,testV,label='test set')


plt.xlabel('n neighbours')
plt.ylabel('tarining/testing  succes')
plt.legend()
plt.show()





