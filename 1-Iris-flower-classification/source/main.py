import pandas as pd
from pandas import read_csv
import numpy as np
import math
import operator
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

url = "../datasets/iris.data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

x_train, x_test = train_test_split(dataset, test_size=0.2)

def euclideanDistance(instance1, instance2, features):
    dist = 0

    for i in range(features):
        dist += np.square(instance1[i] - instance2[i])

    return np.sqrt(dist)

def manhattan_distance(instance1, instance2, features):
    sum = 0
    for i in range(features):
        sum += abs(float(instance1[i]) - float(instance2[i]))

    return sum

def get_neighbors(train, test, k):
    features = test.shape[1] - 1
    numTrain = train.shape[0]
    numTest = test.shape[0]
    num_neighbors = k
    perc_correct = 0

    for i in range(numTest):
        dist_mat = []
        test_row = test.iloc[[i]]
        
        for j in range(numTrain):
            train_row = train.iloc[[j]]

            test_row_tmp = test_row.to_numpy()[0]
            train_row_tmp = train_row.to_numpy()[0]
            train_class = train_row.iloc[0, [-1]].to_numpy()[0]

            dist_mat.append([euclideanDistance(test_row_tmp, train_row_tmp, features), train_class])

        dist_mat = np.array(dist_mat)
        sorted_ind = np.argsort(dist_mat[:,0])[:num_neighbors]

        nearest_neighbors = dist_mat[sorted_ind,:]

        classLabel, frequency = np.unique(nearest_neighbors[:,1], return_counts=True)

        perc_correct = perc_correct + int(test_row.iloc[0, [-1]].to_numpy()[0] == classLabel[np.argmax(frequency)])

    perc_correct = perc_correct*100/numTest

    return perc_correct

get_neighbors(x_train, x_test, round(math.sqrt(x_train.shape[0] + x_test.shape[0])))

def cross_val_knn(dataset, fold, k):
    points = dataset.shape[0]
    bin_sizes = [points // fold + (1 if x < points % fold else 0)  for x in range (fold)]
    sum_score = 0
    
    print("Staring with fold loop")
    for i in range(fold):

        if(not i):
            x_test = dataset[0:(bin_sizes[i])]
            x_train = dataset[(bin_sizes[i]):]

            bin_score = get_neighbors(x_train, x_test, k)
            print(bin_score)
            sum_score = sum_score + bin_score
        else:
            x_test = dataset[sum(bin_sizes[0:i]):(sum(bin_sizes[0:i]) + bin_sizes[i])]
            x_train = pd.concat([dataset[0:sum(bin_sizes[0:i])], dataset[(sum(bin_sizes[0:i]) + bin_sizes[i]):]])

            bin_score = get_neighbors(x_train, x_test, k)
            print(bin_score)
            sum_score = sum_score + bin_score

    return sum_score/fold

Karray= [1,3,11,23,round(math.sqrt(x_train.shape[0] + x_test.shape[0])),57]
for k in Karray:
    print(cross_val_knn(dataset, 5, k))
