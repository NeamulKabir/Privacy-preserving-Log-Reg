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
	temp += 0.005
sz = len(lamLimit)
errorArr = list()

for sit in range(sz):
	counter = 0
	featureList = list()
	labelList = list()

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

	for i in range(featureSize):
		a = randint(0,9)
		if (a<5) and testCount < testSize:
			testList.append(normalizedFeatures[i])
			testLabel.append(labelList[i])
		else:
			trainList.append(normalizedFeatures[i])
			trainLabel.append(labelList[i])

	########################################################################################################################

	logisticRegr = LogisticRegression(penalty = 'l2', C= 0.01)

	logisticRegr.fit(trainList, trainLabel)
	testSample = np.array(testList, dtype = np.float)
	predictions = logisticRegr.predict(testSample)

	############################################################
	weightVector = logisticRegr.coef_
	print weightVector[0][0]
	#print weightVector
	epsilon = 0.001
	lam = lamLimit[sit]
	n = counter
	#eta = np.random.gamma(27, 2/(n*epsilon*lam))				# generating noise vector 

	eta = list()
	for i in range(totalFeatures):
		uni = np.random.uniform(0.0,1.0)
		tempMult = (n*epsilon*lam)
		etaTemp = -2*np.log(2/(tempMult * uni))/tempMult
		eta.append(etaTemp)
	#print eta
	etaArr = np.array(eta, dtype=np.float)

	etaMin = np.amin(etaArr, axis=0)
	etaMax = np.amax(etaArr, axis=0)
	#print etaMax, etaMin
	noiseVector = list()
	for i in range(totalFeatures):
		temp = weightVector[0][i] + (etaArr[i]/etaMin)
		noiseVector.append(temp)
	#print noiseVector


	testPrediction = list()
	for it in range(testSize):
		score = 0
		for i in range(totalFeatures):
			score += noiseVector[i]*testList[it][i]
		sigmoid = 1/(1+np.exp(-score))									# getting prediction for test set
		if sigmoid > 0.5:
			testPrediction.append(1)
		else:
			testPrediction.append(0)

	#print uni
	#eta2 = -n*epsilon*lam*np.log(uni)/2#math.exp(-n*epsilon*lam*uni/2)
	#print eta2
	############################################################

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
	precision,recall =0,0
	temp1 = float((tp+fp))
	if temp1 !=0:
		precision = float(tp)/temp1
	temp2 = float((tp+fn))
	if temp2 != 0:
		recall = float(tp)/temp2
	accuracy = float(float(tp+tn)/float(tp+tn+fp+fn))
	errorArr.append(1-accuracy)

	print testSize, match, mismatch, accuracy
	print ('..............{}................{}..........'.format(mPer, mmPer))

with open('Simple_Algo_Lambda','w') as f_file:
	f_file.write('#Lambda\tError\n');
	temp = len(errorArr)
	for i in range(temp):
		f_file.write('{}\t{}\n'.format(lamLimit[i],errorArr[i]))





















