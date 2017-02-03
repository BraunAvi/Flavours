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

labels = map(int, labels)
print ('lables:', labels)

# Making problem simpler: convert to positive/negative reviews.
labels=list(labels)
answers = (np.array(labels)) >=4
print (answers)


from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB




# Cross validation / overfitting
from sklearn.cross_validation import train_test_split


# take the 100 the most frequent words
for max_features in [2000]:
    vectorizer = CountVectorizer(max_features=max_features)
    vectorizer.fit(reviews)
    counts = vectorizer.transform(reviews).toarray()
    kaggle_test_counts = vectorizer.transform(kaggle_test_reviews).toarray()
    train_counts, test_counts, train_labels, test_labels = train_test_split(counts, answers, random_state=42)
    nb_clf = MultinomialNB()
    nb_clf.fit(train_counts, train_labels)
    print (roc_auc_score(test_labels, nb_clf.predict_proba(test_counts)[:, 1]))

print counts.shape



# Bernoulli:

from sklearn.naive_bayes import BernoulliNB
nb_clf = BernoulliNB()
nb_clf.fit(train_counts, train_labels)
print 'Berbouli: ',roc_auc_score(test_labels, nb_clf.predict_proba(test_counts)[:, 1])


### multinomial
from sklearn.naive_bayes import MultinomialNB
nb_clf = MultinomialNB()
nb_clf.fit(train_counts, train_labels)
print 'multinomial:',roc_auc_score(test_labels, nb_clf.predict_proba(test_counts)[:, 1])
train_counts.shape


## Linear regression + Ridge regularization


from sklearn.linear_model import Ridge
ridge_clf = Ridge()
ridge_clf.fit(train_counts, train_labels)
# use `predict` method for regression methods to evaluate function for new data
print roc_auc_score(test_labels,  ridge_clf.predict(test_counts))
print roc_auc_score(train_labels, ridge_clf.predict(train_counts))
# create_solution(nb_clf.predict_proba(kaggle_test_counts)[:, 1], filename='1-restaurant-predictions-nb.csv')


# Vectoraise te dataset:

maxF=30000
vectorizer_reg = CountVectorizer(max_features=maxF)
vectorizer_reg.fit(reviews)
counts_reg = vectorizer_reg.transform(reviews)
counts_kaggle_reg = vectorizer_reg.transform(kaggle_test_reviews)
counts_kaggle_reg.shape

train_counts_reg, test_counts_reg, train_labels_reg, test_labels_reg = \
    train_test_split(counts_reg, answers, random_state=42, train_size=65000)


# play with regularization here
prediction = []
alphaRange = [0.1, 1, 10, 100, 200, 300, 400, 500, 600, 1000, 10000]
alphaRange = [100, 200, 300, 400, 500, 600, 800]

for alpha in alphaRange:
    ridge_regularization = Ridge(alpha)
    ridge_regularization.fit(train_counts_reg, train_labels_reg)
    prediction.append(roc_auc_score(test_labels_reg, ridge_regularization.predict(test_counts_reg))
                      )

    print 'alpha is:', alpha, 'predict rank:', roc_auc_score(test_labels_reg,
                                                             ridge_regularization.predict(test_counts_reg))
plt.plot(alphaRange,prediction, label='')
plt.xlabel('alpha')
plt.ylabel('testset success')
plt.xscale('log')
# plt.legend()
plt.show()


# find popular words:

dictionary = np.empty(len(vectorizer.vocabulary_), dtype='O')
for word, index in vectorizer.vocabulary_.iteritems():
    dictionary[index] = word


counts_tot=(sum(counts)) # sum over each words
#  remember:   counts = vectorizer.transform(reviews).toarray()

counts_tot_sort=np.array(counts_tot).argsort()[::-1]
import pandas as pd
df=pd.DataFrame({'Word': dictionary[counts_tot_sort[:]],
                 'Frequency':counts_tot[counts_tot_sort[:]]})

print 'The most Frequent word is:',df['Word'][0]

print df




# plt.plot(n,trainV, label='training set')
# plt.plot(n,testV,label='test set')
#
#
# plt.xlabel('n neighbours')
# plt.ylabel('tarining/testing  succes')
# plt.legend()
# plt.show()





