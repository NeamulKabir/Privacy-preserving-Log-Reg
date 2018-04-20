from __future__ import division
from itertools import islice
from random import randint
from random import randint
from sklearn.linear_model import LogisticRegression
import numpy as np
import math

lamLimit = list()
temp = 0.01
for i in range(5):
	lamLimit.append(temp)
	temp += 0.05
sz = len(lamLimit)
errorArr = list()

for sit in range(sz):
	featureList = list()
	labelList = list()
	counter = 0
	with open("1000_test.csv",'r') as f_file:
		next(f_file)
		for line in f_file:
			#print line
			tempList = list()
			data = line.split(',')
			dataSize = len(data)
			for i in range(dataSize):
				if i == 0:
					labelList.append(int(float(data[i])))
				else:
					tempList.append(float(data[i]))

			featureList.append(tempList)
			counter += 1
			if counter == 18000:
				break

	print len(featureList), len(labelList), len(featureList[0])


	featureArr = np.array(featureList, dtype=np.float)

	featureMin = np.amin(featureArr, axis=0)
	featureMax = np.amax(featureArr, axis=0)
	#print('feature min:{}'.format(featureMin))
	#print('feature max:{}'.format(featureMax))

	totalFeatures = len(featureMin)
	#print totalFeatures
	maxValues = list()
	for i in range(totalFeatures):
		temp = abs(featureMin[i])
		if temp > featureMax[i]:
			maxValues.append(temp)
		else:
			maxValues.append(featureMax[i])
	#print maxValues

	normalizedFeatures = list()
	featureSize = len(featureList)
	trainSize = int(featureSize * 80 / 100)
	testSize = featureSize - trainSize
	trainList = list()
	testList = list()
	trainLabel = list()
	testLabel = list()
	testCount = 0

	for item in featureList:
		norm = 0
		for i in range(totalFeatures):
			item[i] = float(item[i])/float(maxValues[i])
			norm += item[i]*item[i]
		normalizedFeatures.append(item)

	#print len(normalizedFeatures), normalizedFeatures[0]
	################################################################################################################################

	for i in range(featureSize):					
		a = randint(0,9)
		if (a<5) and testCount < testSize:
			testList.append(normalizedFeatures[i])					# dividing the features in training and test
			testLabel.append(labelList[i])
		else:
			trainList.append(normalizedFeatures[i])
			trainLabel.append(labelList[i])
	##############################################################################
	learningRate = 0.005							
	regularization = lamLimit[sit]
	featureArr = np.array(trainList, dtype=np.float)
	labelArr = np.array(trainLabel)
	print featureArr.shape[1]

	n = counter
	epsilon = 0.01						######epsilon######
	eta = list()
	for i in range(totalFeatures):
		uni = np.random.uniform(0.0,1.0)
		etaTemp = -2*np.log(2/epsilon)/epsilon
		eta.append(etaTemp)
	#print eta
	etaArr = np.array(eta, dtype=np.float)

	etaMin = np.amin(etaArr, axis=0)
	etaMax = np.amax(etaArr, axis=0)
	#print etaMax, etaMin
	noiseVector = list()
	for i in range(totalFeatures):
		noiseVector.append(-etaArr[i]/etaMin)
	nv = np.array(noiseVector, dtype = np.float)

	weights = np.zeros(featureArr.shape[1])

	for iterator in xrange(30000):										# implementation of log regression
		scores = np.dot(featureArr, weights)
		predictions = 1/(1+np.exp(-scores))

		outputError = labelArr - predictions
		gradient = np.dot(featureArr.T, outputError)
		temp = np.dot(weights,weights)
		temp2 = np.dot(weights, noiseVector)
		weights += (.5* regularization*temp) + (temp2/n) +learningRate * gradient		# need to add noise vector here
	#print temp
	############################################################################
	#print weights



	testPrediction = list()						
	testFeature = np.array(testList, dtype = np.float)
	for item in testFeature:
		score = np.dot(item, weights)
		sigmoid = 1/(1+np.exp(-score))									# getting prediction for test set
		if sigmoid > 0.5:
			testPrediction.append(1)
		else:
			testPrediction.append(0)

	match = 0
	mismatch  = 0

	tp,fp,tn,fn = 0,0,0,0

	for iterate in range(testSize):
		if(testLabel[iterate] == testPrediction[iterate]):
			match += 1
			if(testLabel[iterate] == 0):
				tn += 1
			else:
				tp += 1
		else:
			mismatch += 1
			if(testLabel[iterate] == 1):
				fn +=1
			else:
				fp += 1

	mPer = match * 100 / testSize
	mmPer = mismatch * 100 /testSize
	precision,recall = 0,0
	temp1 = float((tp+fp))
	if temp1 !=0:
		precision = float(tp)/temp1
	temp2 = float((tp+fn))
	if temp2 !=0:
		recall = float(tp)/temp2
	accuracy = float(float(tp+tn)/float(tp+tn+fp+fn))

	print testSize, match, mismatch, accuracy
	print ('..............{}................{}..........'.format(mPer, mmPer))
	er = 1 - accuracy
	errorArr.append(er)

with open('New_Algo_Lambda','w') as f_file:
	f_file.write('#Epsilon\tError\n');
	temp = len(errorArr)
	for i in range(temp):
		f_file.write('{}\t{}\n'.format(lamLimit[i],errorArr[i]))















