# -*- coding: utf-8 -*-
"""
@author: Devansh Gupta
"""

#Kalman - Trilateration

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt, dot, cross                       
from numpy.linalg import norm  
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
    
    Q = 1e-5 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = 0.01 # estimate of measurement variance, change to see effect
    
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
    
    
#Selecting strongest three signals
pos_beacons = [[23,17,2],[23,41,2],[1,15,2],[1,41,2]]

def trilaterate(P1,P2,P3,r1,r2,r3):
    e_x = []
    e_y = []
    temp1=[]
    temp2=[]
    for i in range(3):
        temp1.append(P2[i] - P1[i])                                      
        temp2.append(P3[i] - P1[i])
    e_x = temp1/norm(temp1)                                                               
    i = dot(e_x,temp2)                                   
    temp3 = temp2 - i*e_x                                
    e_y = temp3/norm(temp3)                              
    e_z = cross(e_x,e_y)                                 
    d = norm(temp1)                                      
    j = dot(e_y,temp2)                                   
    x = (r1*r1 - r2*r2 + d*d) / (2*d)                 
    y = (r1*r1 - r3*r3 -2*i*x + i*i + j*j) / (2*j)       
    temp4 = r1*r1 - x*x - y*y    
                     
    if temp4<0:
        return None
    else:
        z = sqrt(temp4)                                      
        p_12_a = P1 + x*e_x + y*e_y + z*e_z                  
        p_12_b = P1 + x*e_x + y*e_y - z*e_z  
        return p_12_a,p_12_b


result_coord = np.full((290,3), 100.0000000)
d=0
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split    
train, test = train_test_split(fp_data, test_size=0.1, random_state = 0)
for row in test:
    p = np.array([row[3],row[4],row[5],row[6]])
    index = np.argsort(p)[:3]

    T = 10.7
    n=1.8
    distance=[]
    for i in range(3):
        distance.append(10**((T+p[index[i]])/(10**n)))


    
    #print (trilaterate(pos_beacons[index[0]],pos_beacons[index[1]],pos_beacons[index[2]],distance[0],distance[1],distance[2]))
    j=0
    while j<1000:
        res = trilaterate(pos_beacons[index[0]],pos_beacons[index[1]],pos_beacons[index[2]],distance[0],distance[1],distance[2])
        if (res == None):
            distance[0]=1.01*distance[0]
            distance[1]=1.01*distance[1]
            distance[2]=1.01*distance[2]
            j=j+1
        else:
            result_coord[d][0] = res[0][0]
            result_coord[d][1] = res[0][1]
            result_coord[d][2] = res[0][2]
            break
    if j==1000:
        result_coord[d][0] = 1000
        result_coord[d][1] = 1000
        result_coord[d][2] = 1000
    d=d+1
    
sum = 0
r=0
#Measuring the accuarcy
for e in range(result_coord.shape[0]):
    if result_coord[e][0]==1000:
        continue
    else:
        sum = sum + math.sqrt((result_coord[e][0] - fp_data[e][0])**2 + (result_coord[e][1] - fp_data[e][1])**2 + (result_coord[e][2] - fp_data[e][2])**2)
        r=r+1
accuracy = sum/r
elapsed_time = process_time() - t
