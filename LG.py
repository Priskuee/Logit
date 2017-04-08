import os
import random
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def get_files(directories):
    files = []
    for d in directories:
        files.extend([os.path.abspath(os.path.join(d, f)) for f in os.listdir(d)])
    return files
def parse_email(fobject):
    email = []
    for line in fobject:
        if line.startswith('Subject:'):
            continue
        else:
            line = line.lower()
            email.extend(re.findall(r'\b[a-z]+\b', line))
    return email

def get_word_bag(files, counter=None):
    if counter is None:
        counter = Counter()
    for f in files:
        counter.update(set(parse_email(open(f,"r",encoding='utf-8', errors='ignore'))))
    return counter

class LgClassifier(object):
    def __init__(self):
        self.w = None
        self.features = None
        
    def trainLG(self, spam_train, ham_train):
        spam_counter = get_word_bag(spam_train)
        ham_counter = get_word_bag(ham_train)
        spam_words = set(spam_counter.keys())
        ham_words = set(ham_counter.keys())
        spam_features = dict((w, spam_counter[w]) for w in list(spam_words - ham_words))
        spam_features = Counter(spam_features)
        self.spam_features = [word for word, _ in spam_features.most_common(20000)]
        
        ham_features = dict((w, ham_counter[w]) for w in list(ham_words - spam_words))
        ham_features = Counter(ham_features)
        self.ham_features = [word for word, _ in ham_features.most_common(20000)]
        self.features = sorted(self.spam_features + self.ham_features)
        X = []
        y = []
        for spam in spam_train:
            words = parse_email(open(spam, "r",encoding='utf-8', errors='ignore'))
            vec = get_vector(self.features, set(words))
            X.append(vec)
            y.append(0)
            
        for ham in ham_train:
            words = parse_email(open(ham, "r",encoding='utf-8', errors='ignore'))
            vec = get_vector(self.features, set(words))
            X.append(vec)
            y.append(1)
            
        self.X = np.matrix(X)
        self.y = np.matrix(y).T 
        
        print('X shape: ', self.X.shape)
        print('y shape: ', self.y.shape)
        
        omega = np.matrix([random.uniform(0.000001) for _ in range(len(self.features))]).T
        print('Omega shape: ', str(len(omega)))
        
        res = minimize(binary_cost, omega, (self.X, self.y), method='BFGS', jac=gradient)
        self.omega = np.matrix(res.x).T
        print('training complete'+str(X))
        
    def classify(self, email):
        words = parse_email(open(email, "r",encoding='utf-8', errors='ignore'))
        vec = np.matrix(get_vector(self.features, set(words))).T
        prob = sigmoid(np.dot(self.omega.T, vec))
        #print(prob)
        if prob >= 0.5:
            return 1
        else:
            return 0

def binary_cost(omega, X, y):
    """
    http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
    """
    X = np.matrix(X)
    y = np.matrix(y)
    omega = np.matrix(omega)
    
    first_bit = np.multiply(-y, np.log(sigmoid( np.dot( X, omega.T ))))
    second_bit = np.multiply(1 - y, np.log(1 - sigmoid( np.dot( X, omega.T ))))
    cost = np.sum(second_bit - first_bit)
    assert cost != np.nan
    return cost

def get_vector(features, bag_of_words):
    vec = []
    for word in features:
        if word in bag_of_words:
            vec.append(1)
        else:
            vec.append(0)
    return np.array(vec)

def gradient(theta, X, y):
    """
    http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad


def sigmoid(a):
    return 1 / (1 + np.exp(-a))
spam_dirs = ['C:/enron11/spam']
spam_files = get_files(spam_dirs)
random.shuffle(spam_files)
split = int(len(spam_files) * 0.7)
spam_train = spam_files[:split] 
spam_test = spam_files[split:]

ham_dirs = ['C:/HWK3_CMPS242/enron11/ham']
ham_files = get_files(ham_dirs)
random.shuffle(ham_files)
ham_train = ham_files[:split] 
split = int(len(ham_files) * 0.7)
ham_test = ham_files[split:]
spam_test_data = [(f, 'spam') for f in spam_test]
ham_test_data = [(f, 'ham') for f in ham_test]

#regr = linear_model.LinearRegression()
#regr.fit(spam_features,ham_features)
lreg = LgClassifier()
lreg.trainLG(spam_train, ham_train)
spamcount=0
hamcount=0
print(lreg.ham_features)
for spam in spam_test:
	result=lreg.classify(spam)
	if result==1:
		spamcount=spamcount+1
	else:
		hamcount=hamcount+1
print(spamcount,'  ',hamcount,' ',str(len(spam_test)))
#for spam in spam_test:
 #   result=lreg.classify(spam)
  #  if(result == 1):
   #  spam=spam+1
    # print('spam')
#print(spam)