import pandas as pd
import numpy as np 
import math
import operator
from timeit import default_timer as timer 
from sklearn.model_selection import train_test_split
from numba import guvectorize, cuda

@guvectorize(['float64(float64, float64)' ], target='cuda')

def MeasureDistance(d1,d2):
#	print(d1, d2)
	k=d1-d2
	s=(k[0]*k[0])+(k[1]*k[1])+(k[2]*k[2])+(k[3]*k[3])
	return s
#Measuring Distance 


def getDistance(train, t, k):
	train['dist']=0.0

	for i in range(len(train)):
		train['dist'][i]=MeasureDistance(np.array(train.iloc[[i],[0,1,2,3]], dtype='float64')[0],np.array(t, dtype='float64'))

# Inserts the distance by column

def getVote(top):
	a={}
#	print(top[4][0])       <---------------------------------------------- Check This
	for i in range(len(top)):
		if(top[4][i] in a):
			a[top[4][i]]+=1
		else :
			a[top[4][i]]=1

	sortedVotes = sorted(a.items(), key=operator.itemgetter(1))
	return sortedVotes[0][0]

#Predicts the class which is closest most number of times

def predict(train_new,test,k):
	test[5]=None
	for i in range(len(test)):
		getDistance(train,test.iloc[i,0:4],k)
		train_sort=train.sort_values(['dist'], ascending=[True])
		top=train_sort.iloc[0:k,]
		test[5][i]=getVote(top.reset_index(drop=True))
#		print(test[5][i])    <---------------------------------------------- Check This

#Saves the predicted class

def accuracy(test):
	l=len(test)
	a=0.0
	for i in range(l):
		if (test[4][i]==test[5][i]):
			a+=1.0
	temp=a*100/l

	print('Accuracy is ',temp, '%')

#Calculating the accuracy

df=pd.read_csv('iris.data', header=None)

#msk=np.random.rand(len(df)) = 0.8
#train=df[msk]
#test=df[~msk]
train, test=train_test_split(df, test_size=0.20 , random_state=1)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Training and Testing Datasets are created

start=timer()
predict(train,test,5)
Time_taken=timer()-start
print(test)
accuracy(test)
print("Time taken to compute is ", Time_taken)
