#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
from patsy import dmatrices
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from operator import itemgetter


################## Data Processing Fuctions Begin ##############################


def preprocessing(file_name):
    with open(file_name) as f:
        z =[]
        for line in f:
            x = [int(i) for i in line.split('\t')]
	    y = [x[0], x[1:]]
            z.append(y)
    return z

def numfollowing(all):
    all_num_followings = []
    for i in training:
        num = len(i[1])
        all_num_followings.append([i[0] ,num])
    return all_num_followings

def numfollowed(id, all):
    count = 0
    for i in all:
        for j in i[1]:
            if(id == j):
		count = count + 1
    return count

def num_allfollowed(all):
    all_num_followed = []
    for i in all:
        all_num_followed.append([i[0], numfollowed(i[0], all)])
    return all_num_followed

    
def num_allfollowed_to_file(all):    
    with open("all_num_followed.txt", "w") as text_file:
        for item in all:
            text_file.write("%s %s\n" % (item[0], item[1]))
def num_followed_from_file(file_name):
    with open(file_name) as f:
        z =[]
        for line in f:
            x = [int(i) for i in line.split(' ')]
	    y = [x[0], x[1]]
            z.append(y)
    return z
def probability_of_friendship(all):
    x = []
    p = 0.0
    for i in all:
#        p = i[1]/float(len(all))
        p = m.sqrt((i[1]/float(len(all))) * 200)
        x.append([i[0], p])
    return x

def write_feature_table_to_file(feature1, feature2):
    with open("training_set_processed.txt", "w") as text_file:
        for i in range(len(feature1)):
            text_file.write("%s %s\n" % (feature1[i][1], feature2[i][1]))

################## Data Processing Fuctions End ##############################


################################################################
#                                                              #   
#   Feature 1: Number of following by a user                   #
#              Score them the following scale                  #
#   0: 1, 1-100: 1, 101-200: 2, 201-400:3, 401-1000: 4, 1001: 5#
#                                                              #
################################################################

training = preprocessing('train.txt')
#Calulate the number of following for a user
all_num_following = numfollowing(training)
#TODO Not for a particular reason to sort them 
all_num_following = sorted(all_num_following, key=itemgetter(0))



for i in range(len(all_num_following)):
    if(all_num_following[i][1] == 0):
	all_num_following[i][1] = 0
    elif(all_num_following[i][1] < 100):
	all_num_following[i][1] = 1
    elif(all_num_following[i][1] > 100 and all_num_following[i][1] < 200):
        all_num_following[i][1] = 2
    elif(all_num_following[i][1] > 200 and all_num_following[i][1] < 400):
	all_num_following[i][1] = 3
    elif(all_num_following[i][1] > 400 and all_num_following[i][1] < 1000):
	all_num_following[i][1] = 4
    else:
	all_num_following[i][1] = 5
#print all_num_following





################################################################
#                                                              #   
#   Feature 2: Number of followed for a user                   #
#              Score them the following scale                  #
#   0: 1, 1-100: 1, 101-200: 2, 201-400:3, 401-1000: 4, 1001: 5#
#                                                              #
################################################################


#Calulate the number of followed for a user
all_num_followed = num_followed_from_file('all_num_followed.txt')
#TODO Not for a particular reason to sort them 
all_num_followed = sorted(all_num_followed, key=itemgetter(0))


for i in range(len(all_num_followed)):
    if(all_num_followed[i][1] == 0):
	all_num_followed[i][1] = 0
    elif(all_num_followed[i][1] < 100):
	all_num_followed[i][1] = 1
    elif(all_num_followed[i][1] > 100 and all_num_followed[i][1] < 200):
        all_num_followed[i][1] = 2
    elif(all_num_followed[i][1] > 200 and all_num_followed[i][1] < 400):
	all_num_followed[i][1] = 3
    elif(all_num_followed[i][1] > 400 and all_num_followed[i][1] < 1000):
	all_num_followed[i][1] = 4
    else:
	all_num_followed[i][1] = 5
   
#print all_num_followed

################################################
#                                              #
#   Feature 3: Positive link & Nagative link in#
#              the training set for an ID      #
#                                              #
################################################
#print training

#for i in training:
#   for j in i[i][1]:
#       if(j):


################################################
#                                              #
#   Feature 3: The probability of friendship   #
#              in the dataset                  #
#                                              #
################################################


#Calulate the probability of having friendship among users and it is sorted as well
#all_probability_friendship = probability_of_friendship(all_num_followings)





################################################
#                                              #
#   Constructing the feature table             #
#                                              #
################################################

#write_feature_table_to_file(all_num_following, all_num_followed, all_probability_friendship)
write_feature_table_to_file(all_num_following, all_num_followed)
# Use panda in-memory power to speed up 
train_data = pd.read_csv('training_set_processed.txt', sep=" ", header = None)
train_data.columns = ["Following", "Followed"]
#train_data.columns = ["Following", "Followed", "Probability_of_Friendship"]
#train_data['friendship'] = (train_data.Following > 1000).astype(int)
train_data['friendship'] = (train_data.Following >= 5).astype(int)
#print train_data.groupby('Following').mean()
#print train_data
#y, X = dmatrices('friendship ~ Following + Followed + Probability_of_Friendship', train_data, return_type="dataframe")
y, X = dmatrices('friendship ~ Following + Followed', train_data, return_type="dataframe")

# flatten y into a 1-D array
y = np.ravel(y)
#print y

################################################
#                                              #   
#   Training Alogrithm                         #
#                                              #
################################################

#clf_l1_LR = LogisticRegression()

#clf_l1_LR.fit(X, y)
#print clf_l1_LR.score(X, y)
#print y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
predicted = model2.predict(X_test)
print predicted
probs = model2.predict_proba(X_test)
print probs
#print metrics.accuracy_score(y_test, predicted)
#print metrics.roc_auc_score(y_test, probs[:, 1])

#model2.predict_proba(np.array([1,2, 0.3]))
 
