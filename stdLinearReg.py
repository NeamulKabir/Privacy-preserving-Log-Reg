from __future__ import division
from itertools import islice
from random import randint
from random import randint
from sklearn.linear_model import LogisticRegression
import numpy as np
import math


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
		if counter == 10000:
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

logisticRegr = LogisticRegression(penalty = 'l2', C= 0.01)		# regularization parameter included

logisticRegr.fit(trainList, trainLabel)
testSample = np.array(testList)
predictions = logisticRegr.predict(testSample)

match = 0
mismatch  = 0

tp,fp,tn,fn = 0,0,0,0

for iterate in range(testSize):
	if(testLabel[iterate] == predictions[iterate]):
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
precision = float(tp)/float((tp+fp))
recall = float(tp)/float((tp+fn))
accuracy = float(float(tp+tn)/float(tp+tn+fp+fn))
f1_score = 2*precision*recall / (precision+recall)

print testSize, match, mPer, mismatch, mmPer, precision, recall, accuracy, f1_score




















