import numpy as np;
import matplotlib.pyplot as plt
import networkx as nx;
from numpy import linalg as LA

#define Array and Graph
A = np.array([[0,1,1,0,1],[1,0,0,0,0],[1,0,0,1,0],[0,0,1,0,1],[1,0,0,1,0]]);A = np.array([[0,1,1,1,0,0,0,0,0],[1,0,1,0,0,0,0,0,0],[1,1,0,1,0,0,0,0,0],[1,0,1,0,1,1,0,0,0],[0,0,0,1,0,1,1,1,0],
             [0,0,0,1,1,0,1,1,0],[0,0,0,0,1,1,0,1,1],[0,0,0,0,1,1,1,0,0],[0,0,0,0,0,0,1,0,0]])
H = nx.from_numpy_matrix(A);


#function to calculate degree centrality
def degreeCentrality(A):
     sum = 0;
     result = [];
     
     rows = A.shape[0];
     cols = A.shape[1];
     
     for i in range(0, rows):
         #print(row);
         sum = 0;
         for j in range(0, cols):
             #print(col);
             if i != j:
                 sum += A[i][j];
         result.append(sum / (rows - 1));
     return result;


#function to calculate closeness centrality
def closenessCentrality(A):  
    H = nx.from_numpy_matrix(A);
    length = list(nx.all_pairs_shortest_path_length(H));
    print(length)
    distanceMatrix = [];
    rows = len(length);
    for i in range(0, rows):
        x = length[i];
        y = x[1];
        for j in range(0, rows):
            distanceMatrix.append(y[j]);
      
    a = np.array(distanceMatrix);
    a = a.reshape(rows, rows);
    sum = 0;
    result1 = [];
    rows = a.shape[0];
    cols = a.shape[1];
    for r in range(0, rows):
        sum = 0;
        for c in range(0, cols):
            if(r != c):
                sum += a[r][c];
        result1.append((rows - 1) / sum);
    return result1   
    
#Betweeness centrality
def calcSigma(A):
    result = [];
    powerMatrix = [np.linalg.matrix_power(A,m) for m in [1,2,3,4,5]]
    totalSum = np.zeros((A.shape[0],A.shape[1]))
    min_lengths= np.zeros((A.shape[0],A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for x in range(5):
                X = powerMatrix[x].copy()
                if X[i,j] > 0 and i != j:
                    totalSum[i,j] = X[i,j]
                    min_lengths[i,j] = x + 1
                    break
    result.append(totalSum)
    result.append(min_lengths)
    return result

def sigma_through_jay(j):
    values = calcSigma(A)
    totalSum = values[0].reshape((A.shape[0],A.shape[1]))
    min_lengths = values[1]
    B = A.copy()
    B[j] = np.zeros(A.shape[1])
    B[:,j] = np.zeros(A.shape[1])
    B_powers = [np.linalg.matrix_power(B,m) for m in [1,2,3,4,5]]
    B_sigma = np.zeros((A.shape[0],A.shape[1]))
    for i in range(9):
        for k in range(9):
            if i != k:
                X = B_powers[int(min_lengths[i,k]-1)]
                B_sigma[i,k] = X[i,k]
    return (totalSum - B_sigma)      

def betweenness(j):
    values = calcSigma(A)
    totalSum = values[0].reshape((A.shape[0],A.shape[1]))
    sigst = sigma_through_jay(j)
    total = 0
    for i in range(9):
        for k in range(i):
            if i != j and k != j:
                total += sigst[i,k] / totalSum[i,k]
    
    return total
    

#Function to calculate Eigen vector centrality
 def eigenCentrality(Adjmat):
    w, v = LA.eig(np.matrix(Adjmat))
    #Largest eigen value
    max = w[0];
    index = 0
    for i in range(1, len(w)):
        if(max < w[i]):
            max = w[i]
            index = i
    #print eigen vector corresponding to largest value        
    res = v[:, index]
    return res  
