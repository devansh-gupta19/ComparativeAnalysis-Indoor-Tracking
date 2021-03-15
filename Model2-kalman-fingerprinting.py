# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 00:39:18 2020

@author: Devansh Gupta
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
from time import process_time

t = process_time()
#importing the dataset
dataset = pd.read_csv('data_rssi.csv')
dataset['beacon'] = dataset['ap']
X = dataset.iloc[:, 1].values

#Generating dictionarry of dataframes having same object coordinates and ap
dfs={}
result = {}

for k, d in dataset.groupby(['ap','x','y','z']):
    dfs[k]=d
    
for key in dfs.keys():     
    m = np.asarray(dfs[key]['signal'])
    
    plt.rcParams['figure.figsize'] = (10, 8)
    
    # intial parameters
    n_iter = m.shape[0]
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    #z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
    
    Q =1e-2 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 0.15 # estimate of measurement variance, change to see effect
    
    # intial guesses
    xhat[0] = 12.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(m[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k] 
    result[key] = np.nanmean(xhat[1:])  


result_data = pd.Series(result).rename_axis(['AP', 'x','y','z']).reset_index(name='RSSI')
result_data['RSSI'].fillna(0, inplace=True)
coord = {}
for k, d in result_data.groupby(['x','y','z']):
    coord[k]=d

  
#Creating Fingerprinting Database
fp_data = np.full((290,7), 100)

j=0
for key in coord:
    data = coord[key]
    for i in range(data.shape[0]):
        if (data['AP'].values[i] == 'A'):
            fp_data[j][3] = data['RSSI'].values[i]
        elif (data['AP'].values[i] == 'B'):
            fp_data[j][4] = data['RSSI'].values[i]
        elif (data['AP'].values[i] == 'C'):
            fp_data[j][5] = data['RSSI'].values[i]
        elif (data['AP'].values[i] == 'D'):
            fp_data[j][6] = data['RSSI'].values[i]
        fp_data[j][0] = data['x'].values[i]
        fp_data[j][1] = data['y'].values[i]
        fp_data[j][2] = data['z'].values[i]
    j=j+1


#fp_data = shuffle(fp_data)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split    
train, test = train_test_split(fp_data, test_size=0.1, random_state = 0)

#Online Phase (WKNN)
j=0
x=[0]*test.shape[0]
y=[0]*test.shape[0]
z=[0]*test.shape[0]
for row in test:
    p = (row[3],row[4],row[5],row[6])
    k=4
    distance=[]

    for i in range(fp_data.shape[0]):    
        #calculate the euclidean distance of p from training points  
        euclidean_distance = math.sqrt((fp_data[i][3]-p[0])**2 + (fp_data[i][4]-p[1])**2 + (fp_data[i][5]-p[2])**2 + (fp_data[i][6]-p[3])**2)
        distance.append((euclidean_distance,(fp_data[i][0],fp_data[i][1],fp_data[i][2])))

    #Selecting k nearest neighbours       
    distance = sorted(distance)[:k]

    #Calculating weights
    weights=[] 
    total=0
    for i in range(len(distance)): 
        total = total + distance[i][0]

    for i in range(len(distance)):
        weights.append(distance[i][0]/total)

    #Predicting object coordinates
    for i in range(k):
        x[j] = x[j] + weights[i] * distance[i][1][0]
        y[j] = y[j] + weights[i] * distance[i][1][1]
        z[j] = z[j] + weights[i] * distance[i][1][2]
    j=j+1
    
sum = 0
#Measuring the accuarcy
for i in range(test.shape[0]):
    sum = sum + math.sqrt((x[i] - test[i][0])**2 + (y[i] - test[i][1])**2 + (z[i] - test[i][2])**2)
accuracy = sum/test.shape[0]
elapsed_time = process_time() - t