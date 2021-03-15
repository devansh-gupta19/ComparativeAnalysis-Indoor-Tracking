# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:25:27 2020

@author: Devansh Gupta
"""
#Gaussian - Moving average filter - Fingerprinting

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.utils import shuffle
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

#Applying gaussain filter to each dataframe in the dictionary
for key in dfs.keys():
    data =dfs[key]
    rssi=data.iloc[:, 1].values

    mean=np.nanmean(dfs[key]['signal'])
    var = np.nanvar(dfs[key]['signal'])
    std = np.nanstd(dfs[key]['signal'])

    gauss = (1/(np.sqrt(var*2*np.pi)))*np.exp(-((rssi-mean)*(rssi-mean))/var)
    
    #creating list of rssi values which lie close to the mean
    c=[]
    for i in dfs[key]['signal']:
        if i>=mean-std and i<=mean+std:
            c.append(i)
            
#storing the values after applyong gaussian filter into a dictionary
    result[key] = c
                       
#Moving Average Filter
result_maf = {}
for key in result.keys():
    values = result[key]
    result_maf[key] = np.nanmean(values)
    
result_data = pd.Series(result_maf).rename_axis(['AP', 'x','y','z']).reset_index(name='RSSI')


  
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