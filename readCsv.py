from __future__ import division
from itertools import islice
from random import randint
from sklearn.linear_model import LogisticRegression
import numpy as np
import math

counter = 1
featureTrainList = list()
labelTrainList = list()

with open("1000_test.csv",'r') as f_file:
	next(f_file)
	for line in f_file:
		#print line
		tempList = list()
		data = line.split(',')
		dataSize = len(data)
		for i in range(dataSize):
			if i == 0:
				labelTrainList.append(int(float(data[i])))
			else:
				tempList.append(float(data[i]))

		featureTrainList.append(tempList)
		counter += 1
		#if counter == 1000:
		#	break

print len(featureTrainList), len(labelTrainList), len(featureTrainList[0])
trainingSize = len(featureTrainList)

featureArr = np.array(featureTrainList, dtype=np.float)

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
print maxValues

filteredFeatures = list()

for item in featureTrainList:
	norm = 0
	for i in range(totalFeatures):
		item[i] = float(item[i])/float(maxValues[i])
		norm += item[i]*item[i]
	#print norm
	if(math.sqrt(norm) <= 1):
		filteredFeatures.append(item)

print len(filteredFeatures)



















