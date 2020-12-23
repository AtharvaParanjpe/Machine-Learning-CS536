import numpy as np
import pandas as pd
from numpy import linalg as LA
import math

m = 4
t = 0


################## Part 1: ######################
def generate_data(m):
    columns = []
    for i in range(2):
        columns.append("x"+str(i+1))
    columns.append("y")

    data = [[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]]
    return pd.DataFrame(data=data, columns=columns)
    
def generate_alphas(m):
    alpha = np.random.rand(m,1)
    return alpha

def kernelValue(x1,x2):
    return (1+np.dot(x1,x2))**2

def computeDeltaF(x,y,alpha, i):
    sum_alpha_y = 0
    for j in range(1,len(alpha)):
        sum_alpha_y+= alpha[j]*y[j]

    sum_alpha_y_x = 0
    for j in range(1,len(alpha)):
        sum_alpha_y_x+= alpha[j]*y[j]*kernelValue(x[j].T,x[0])
    
    sum_alpha_yi_yj = 0
    for j in range(1,len(alpha)):
        if(j!=i):
            sum_alpha_yi_yj+= alpha[j]*y[i]*y[j]*kernelValue(x[i].T,x[j]) 
    
    del_f = -1*y[i]*y[0] + 1 - (sum_alpha_y*y[i]*kernelValue(x[0].T,x[0])) + (sum_alpha_y*y[i]*kernelValue(x[i].T,x[0])) + (sum_alpha_y_x)*y[i]
    del_f = del_f - sum_alpha_yi_yj - alpha[i]*kernelValue(x[i].T, x[i])
    return del_f


def train_alpha(alpha, x,y):
    
    lr = 0.0001
    t=0

    epsilon_t = np.exp(-4*t)
    while(epsilon_t>0):
        epsilon_t = np.exp(-4*t)
        delta_array = []
        sum_alpha_y = 0
        for j in range(1,len(alpha)):
            sum_alpha_y+= alpha[j]*y[j]
        
        for i in range(1,len(alpha)):
            current_alpha = alpha[i]
            delta = lr*(computeDeltaF(x,y,alpha,i) + epsilon_t*(1/alpha[i] + (y[i])/sum_alpha_y))
            if(current_alpha+delta<0):
                _lr =_lr/5
                i=i-1
            delta_array.append(delta) 
        alpha[1:] = alpha[1:] + delta_array
        t=t+1
    alpha[0] =  0
    for i in range(1,len(alpha)):
        alpha[0]+= -1*alpha[i]*y[i]*y[0]
    return alpha


################### Part 2 : Generating xor data and fitting the svm to it ##################
df = generate_data(m)
print(df)

alpha = generate_alphas(m)
y = df['y'].values
x = df.drop('y',axis=1).values

trainedAlphas = []
for i in range(100):
    trainedAlphas.append(train_alpha(alpha,x,y))

trainedAlphas = np.average(trainedAlphas, axis =0)

print()
print("Trained Alphas: ")
print(trainedAlphas.reshape(1,4))
print()



newSpaceX = pd.DataFrame(data = [1]*df.shape[0], columns= ['constant'])
newSpaceX['x1'] = df['x1']*math.sqrt(2) 
newSpaceX['x2'] = df['x2']*math.sqrt(2) 
newSpaceX['x1x2'] = df['x1']*df['x2']*math.sqrt(2)
newSpaceX['x1^2'] = df['x1']**2
newSpaceX['x2^2'] = df['x2']**2
newSpaceX['y'] = df['y']
print("New Feature Space")
print(newSpaceX)
print()

x = newSpaceX.drop(['y'],axis=1).values
y = newSpaceX['y'].values
weightsOfPrimalSvm = []

##################### Part 3 : Reconstruction of the Primal SVM ######################
primalWeights = 0
for i in range(m):
    primalWeights+= trainedAlphas[i]*y[i]*x[i,:]
print()
print("Primal Weights are :")
print( primalWeights)

#### classification of the XOR data ####

threshold = 0
y_pred = np.dot(x,primalWeights)
newSpaceX['y_pred'] = y_pred
newSpaceX['y_pred'] = np.where(newSpaceX['y_pred']< threshold,-1,1)
y_pred = newSpaceX['y_pred'].values
print()
print("Output with Predictions")
print(newSpaceX)

err = np.average(np.square(y_pred-y))
print()
print("Classification Error is : ", err)



