import csv
import random
import math
import operator
import numpy as np
from sklearn.metrics import f1_score
import statistics
import pandas as pd
from collections import OrderedDict


##############################  GLOBAL FUNCTION  #################################
# transform dataset to split into training set and test set
def loadDataset(X_train, X_test, y_train, y_test, trainingSet=[] , testSet=[]):
    for i in range(len(X_train)):
        row = np.append(X_train[i], y_train[i])
        trainingSet.append(row)

    for i in range(len(X_test)):
        row = np.append(X_test[i], y_test[i])
        testSet.append(row)

# distance metric Euclidean
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# get k nearest neighbor from query testInstance
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1

	if(len(trainingSet) > 0):
		for x in range(len(trainingSet)):
			dist = euclideanDistance(testInstance, trainingSet[x], length)
			distances.append((trainingSet[x], dist))
		distances.sort(key=takeSecond)
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])

		return neighbors
	else:
		return []

# label prediction using majority voting
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]

		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=takeSecond, reverse=True)

	if(sortedVotes):
		label = sortedVotes[0][0] 
		return label
	else:
		return None

# determine error rate
def getErrorRate(testSet, predictions):
	wrongs = 0
	for idx in range(len(testSet)):
		if testSet[idx][-1] != predictions[idx]:
			wrongs += 1

	return (wrongs/float(len(testSet))) * 100.0

# determine F-measure score
def getFScore(testSet, predictions):
	actual = []
	for label in (testSet):
		actual.append(label[-1])

	return f1_score(actual, predictions, average='macro')

# remove array element
def removearray(List_arr, arr):
    ind = 0
    size = len(List_arr)
    while ind != size and not np.array_equal(List_arr[ind],arr):
        ind += 1
    if ind != size:
        List_arr.pop(ind)
    else:
		raise ValueError('array not found in list.')

# get secod element
def takeSecond(elem):
    return elem[1]

############################################################################



############################################################################
##############################  MAIN PROGRAM  ##############################
############################################################################


# ******************************* KNN 1967 *********************************
# Cover; Hart
def knn(k, X_train, X_test, y_train, y_test):
	# prepare data
	trainingSet=[]
	testSet=[]
	
	loadDataset(X_train, X_test, y_train, y_test, trainingSet, testSet)
	predictions=[]

    # test all test set instance
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)

    # evaluation
	error_rate = getErrorRate(testSet, predictions)
	f_score = getFScore(testSet, predictions)
	return error_rate, f_score



# ****************************** WKNN 1976 ********************************
# S. Dudani
# The Distance-Weighted k-Nearest-Neighbor Rule
# IEEE Transactions on Systems, Man and Cybernetics

def wknn(k, X_train, X_test, y_train, y_test):
	# prepare data
	trainingSet=[]
	testSet=[]
	
	loadDataset(X_train, X_test, y_train, y_test, trainingSet, testSet)
	predictions=[]

    # test all test set instance
	for x in range(len(testSet)):
		neighbors = getNeighborsWKNN(trainingSet, testSet[x], k)
		result = getResponseWKNN(neighbors)
		predictions.append(result)

    # evaluation
	error_rate = getErrorRate(testSet, predictions)
	f_score = getFScore(testSet, predictions)
	return error_rate, f_score

# get neighbors
def getNeighborsWKNN(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=takeSecond)

	neighbors = []
	for x in range(k):
		neighbors.append(distances[x])

	return neighbors

# label prediction using weighted distance
def getResponseWKNN(neighbors):
	neighbors.sort(key=takeSecond)

	classVotes 	= {}
	min_dist	= neighbors[0][-1]
	max_dist	= neighbors[-1][-1]

	for x in range(len(neighbors)):
		curr_dist	= neighbors[x][-1]
		label 		= neighbors[x][0][-1]
		
		if(curr_dist == min_dist):
			weight	= 1
		else:
			weight	= float(max_dist - curr_dist) / float(max_dist - min_dist)

		if label in classVotes:
			classVotes[label] += weight
		else:
			classVotes[label] = weight
	sortedVotes = sorted(classVotes.iteritems(), key=takeSecond, reverse=True)

	if(sortedVotes):
		return sortedVotes[0][0]



# ******************************* LMKNN 2006 ******************************
# Mitani, Y., Hamamoto, Y.
# A local mean-based nonparametric classifier
# Pattern Recognition Letters

def lmknn(k, X_train, X_test, y_train, y_test):
	# prepare data
	trainingSet=[]
	testSet=[]
	
	loadDataset(X_train, X_test, y_train, y_test, trainingSet, testSet)

	# generate predictions
	predictions=[]

    # test all test set instance
	for x in range(len(testSet)):
		neighbors 	= getNeighborsLM(trainingSet, testSet[x], k)
		result 		= getResponseLM(k, neighbors)
		predictions.append(result)

    # evaluation
	error_rate = getErrorRate(testSet, predictions)
	f_score = getFScore(testSet, predictions)
	return error_rate, f_score

# get neighbors
def getNeighborsLM(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=takeSecond)

	classes = [d[0][-1] for d in distances]
	classes = list(set(classes))

	neighbors = []
	for c in classes:
		data_class = [dt[0] for dt in distances if dt[0][-1] == c]

		for x in range(k):
			if(x < len(data_class)):
				neighbors.append(data_class[x])

	return neighbors, testInstance

# label prediction using local mean vector
def getResponseLM(k, neighbors):
	classVotes = {}
	neighbors, test = neighbors

	for x in range(len(neighbors)):
		label = neighbors[x][-1]
		data = neighbors[x][:-1]
		
		# select k-nn data by class
		if label in classVotes:
			classVotes[label] = np.sum([classVotes[label], data], axis=0)
		else:
			classVotes[label] = data

	for c in classVotes.keys():
		# local mean
		mean = classVotes[c] / k

		# subtraction
		sub = np.subtract(test[:-1], mean)
		classVotes[c] = np.sum(sub * np.transpose(sub))

	sortedVotes = sorted(classVotes.iteritems(), key=takeSecond)
	return sortedVotes[0][0]


# **************************** PNN 2009 ***********************************
# Zeng, Yong; Yang, Yupu; Zhao, Liang
# Pseudo nearest neighbor rule for pattern classification
# Expert Systems with Applications
def pnn(k, X_train, X_test, y_train, y_test):
	# prepare data
	trainingSet=[]
	testSet=[]
	
	loadDataset(X_train, X_test, y_train, y_test, trainingSet, testSet)
	# generate predictions
	predictions=[]

    # test all test set instance
	for x in range(len(testSet)):
		neighbors 	= getNeighborsPNN(trainingSet, testSet[x], k)
		result 		= getResponsePNN(neighbors)
		predictions.append(result)

    # evaluation
	error_rate = getErrorRate(testSet, predictions)
	f_score = getFScore(testSet, predictions)
	return error_rate, f_score

# get neighbors
def getNeighborsPNN(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=takeSecond)

	classes = [d[0][-1] for d in distances]
	classes = list(set(classes))

	neighbors = []
	for c in classes:
		data_class = [[dt[1], c] for dt in distances if dt[0][-1] == c]

		for x in range(k):
			if(x < len(data_class)):
				neighbors.append(data_class[x])
			
	return neighbors

# label prediction using distance and weight
def getResponsePNN(neighbors, power=1):
	classVotes = {}
	iterator = {}

	for x in range(len(neighbors)):
		currClass = neighbors[x][-1]

		if currClass in classVotes:
			iterator[currClass]	+= 1
			classVotes[currClass] += (neighbors[x][0] / iterator[currClass])
		else:
			iterator[currClass]	= 1
			classVotes[currClass] = neighbors[x][0]

	sortedVotes = sorted(classVotes.iteritems(), key=takeSecond)
	return sortedVotes[0][0]



# ***************************** MKNN 2012 *******************************
# Liu, Huawen; Zhang, Shichao;
# Noisy data elimination using mutual k-nearest neighbor for classification mining
# Journal of Systems and Software
def mknn(k, X_train, X_test, y_train, y_test):
	# prepare data
	trainingSet=[]
	testSet=[]
	
	loadDataset(X_train, X_test, y_train, y_test, trainingSet, testSet)

	# outlier removal
	trainingSet = mknn_outlier_removal(trainingSet, k)

	# MKNNC
	predictions = []
	
	for x in range(len(testSet)):
		candidate = []
		train = trainingSet[:]
		mutuals = getNeighborsMKNN(train, testSet[x], k)

		# if testSet[x] is mutual of NN(x)
		if(len(mutuals) > 0):
			candidate = candidate + mutuals
			label = getResponse(candidate)
		else:
			label = random.choice(y_train)
		
		predictions.append(label)

    # evaluation
	error_rate = getErrorRate(testSet, predictions)
	f_score = getFScore(testSet, predictions)
	return error_rate, f_score

#outlier removal on preprocessing
def mknn_outlier_removal(trainingSet, k):
	ts1 = trainingSet[:]

	for x in trainingSet:
		removearray(ts1, x)
		nn = getNeighborsMKNN(ts1, x, k)

		# if has mutual neighbor(s), return back x to dataset
		if(len(nn) > 0):
			ts1.append((x))

	return ts1

# get neighbors
def getNeighborsMKNN(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=takeSecond)
	neighbors = []

	# main neighbors
	for x in range(k):
		neighbors.append(distances[x][0])

	# mutual neighbors
	mutuals = []
	testlist = testInstance.tolist()

	for n in neighbors:
		train = []
		train = trainingSet[:]
		train.append(testlist)
		train = [ dta for dta in train if not np.array_equal(dta, n) ]
		
		neighs_sub = getNeighbors(train, n, k)
		cek = [ dt for dt in neighs_sub if np.array_equal(dt, testlist)]

		if(len(cek) > 0):
			mutuals = mutuals + [n]

	return mutuals
