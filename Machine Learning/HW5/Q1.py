import numpy as np
import pandas as pd
import math
from numpy.linalg import inv

import matplotlib.pyplot as plt


m=1000
sigma = math.sqrt(0.1)

def generate_data(m):
    global sigma

    X = []
    for i in range(m):
        x_vals = []
        y = 10
        for i in range(20):
            if(i==10):
                r = np.random.normal(0,sigma)
                x = x_vals[0]+x_vals[1] + r
                x_vals.append(x)
            elif(i==11):
                r = np.random.normal(0,sigma)
                x = x_vals[2]+x_vals[3] + r
                x_vals.append(x)
            elif(i==12):
                r = np.random.normal(0,sigma)
                x = x_vals[3]+x_vals[4] + r
                x_vals.append(x)
            elif(i==13):
                r = np.random.normal(0,sigma)
                x = 0.1*x_vals[6] + r
                x_vals.append(x)
            elif(i==14):
                r = np.random.normal(0,sigma)
                x = 2*x_vals[1] - 10 + r
                x_vals.append(x)
            else:
                randomVar = np.random.normal(0,1)
                x_vals.append(randomVar)
            if(i<10):
                y+= x_vals[i]*0.6**(i+1)
        r = np.random.normal(0,sigma)
        y+= r
        x_vals.append(y)
        x_vals=[1] + x_vals
        X.append(x_vals)
    columns = ["b"]
    for i in range(1,21):
        columns.append("x"+str(i))
    columns.append("y")
    df = pd.DataFrame(data=X,columns=columns)
    return df

def compute_w(df):
    y = np.array(df['y'].values)
    y = y.reshape(y.shape[0],1)
    df = df.drop(['y'], axis =1)
    x = np.array(df.values)
    w_prime = np.dot(np.dot(inv(np.dot(np.transpose(x),x)),np.transpose(x)),y)
    return w_prime

def compute_actual_weights():
    actual_weights = [10]
    for i in range(1,11):
        actual_weights.append(0.6**i)
    for i in range(10):
        actual_weights.append(0)
    return actual_weights

def predict(weights,df):
    df = df.drop('y', axis=1)
    y = np.dot(df.values,weights)
    return y

def computeError(y_true, y_pred):
    total = 0
    for x,y in zip(y_true,y_pred):
        total+=(x-y)**2
    return float(total/len(y_true))

############## Q1 Part 1 : 1st term includes bias ##############
df = generate_data(m)
df_test = generate_data(10000)

err = 0
for i in range(50):
    weights = compute_w(df)
    weights = weights.reshape(weights.shape[1],weights.shape[0])[0]
    # print(weights)
    # print()
    y_pred = predict(weights, df_test)
    err += computeError(df_test['y'].values , y_pred)
err = err/50
# print()
print(err)


################ Q1 Part 2 : Including the Ridge regression ################

def compute_ridge_w(df, l):
    y = np.array(df['y'].values)
    y = y.reshape(y.shape[0],1)
    df = df.drop(['y'], axis =1)
    x = np.array(df.values)
    I = np.eye(x.shape[1])
    w_prime = np.dot(np.dot(inv(np.dot(np.transpose(x),x)-l*I),np.transpose(x)),y)
    return w_prime

# df = generate_data(m)
errorWeightArray = []
L = []
for l in range(0,700,1):
    weights = compute_ridge_w(df,float(l/10000))
    weights = weights.reshape(weights.shape[1],weights.shape[0])[0]
    y_pred = predict(weights, df_test)
    err = computeError(df_test['y'].values , y_pred)
    errorWeightArray.append(err)
    L.append(float(l/10000))
    
# weights = compute_ridge_w(df,0.015)
# weights.reshape(weights.shape[1],weights.shape[0])[0]
# # print(weights.reshape(1,21))

# y_pred = predict(weights, df_test)
# err = computeError(df_test['y'].values , y_pred)
# # print(err)

# plt.plot(L,errorWeightArray)
# plt.xlabel('lambda')
# plt.ylabel('error')
# plt.show()


#################Q1 Part 3 ###################

def compute_lasso_w(df,l):
    y = np.array(df['y'].values)
    y = y.reshape(y.shape[0],1)
    df = df.drop(['y'], axis =1)
    x = np.array(df.values)
    w = np.zeros((21,1))
    for j in range(20):
        w[0] = w[0] + np.average(y-np.dot(x, w))
        for i in range(1, w.shape[0]):
            xi = x[:,i].reshape(len(x),1)
            a = (-1*np.dot(np.transpose(xi),y-np.dot(x,w)) + l/2)/np.dot(np.transpose(xi),xi)
            b = (-1*np.dot(np.transpose(xi),y-np.dot(x,w)) - l/2)/np.dot(np.transpose(xi),xi)
            if(w[i]>a):
                w[i] = w[i]-a
            elif(w[i]<b):
                w[i] = w[i]-b
            else:
                w[i] = 0
    return w


# df = generate_data(m)

# errorMean = []
# L = []
# count_noise = []
# iterations = 25
# df_test = generate_data(2000)
# actual_weights = compute_actual_weights()
# for l in np.arange(0,4000,50):
#     weights = None
#     err = 0
#     count = 0
#     for j in range(iterations):
#         weights = compute_lasso_w(df,l)
#         weights = weights.reshape(weights.shape[1],weights.shape[0])[0]
#         count+= np.count_nonzero(weights)
#         y_pred = predict(weights, df_test)
#         err += computeError(df_test['y'].values , y_pred)
#     errorMean.append(err/iterations)
#     count = float(count/iterations)
#     count_noise.append(count)
#     L.append(l)
    
# plt.plot(L,count_noise)
# plt.xlabel('Lambda')
# plt.ylabel('Number of non zero variables')
# plt.show()

############### Part 4 : Optimal Lambda for Error #####################

# plt.plot(L,errorMean)
# plt.xlabel('Lambda')
# plt.ylabel('Error')
# plt.show()
# weights = []

weights = compute_lasso_w(df,5)
weights = weights.reshape(weights.shape[1],weights.shape[0])[0]
print("Weights for optimum lambda: ")
print(weights)
y_pred = predict(weights, df_test)
err = computeError(df_test['y'].values , y_pred)
print("Error for Lasso is : ", err)


################### Part 5 : Take the most significant features and apply ridge regression ###############


## Prune the zero value features
def modify_data(weights):
    pos = np.argwhere(weights > 0)
    pos = pos.reshape(1,pos.shape[0]).tolist()[0]
    new_weights = np.take(weights,pos)
    new_x = []
    columns = ["b"]
    for i in range(1,21):
        columns.append("x"+str(i))
    columns = np.take(columns,pos)
    columns = np.append(columns,['y'])
    return columns, new_weights


columns, new_weights = modify_data(weights)
updateddf = df[columns]

errorArray = []
L = []

df_test = df_test[columns]

for l in range(0,700,5):
    err = 0
    for j in range(50):
        weights = compute_ridge_w(updateddf,float(l/10000))
        weights = weights.reshape(weights.shape[1],weights.shape[0])[0]
        y_pred = predict(weights, df_test)
        err += computeError(df_test['y'].values , y_pred)
    errorArray.append(err/50)
    L.append(float(l/10000))

plt.plot(L,errorArray)
plt.xlabel('lambda')
plt.ylabel('error')
plt.show()

######## Part 5 : Error at optimal lambda ##########
weights = compute_ridge_w(updateddf,0.033)
weights = weights.reshape(weights.shape[1],weights.shape[0])[0]
y_pred = predict(weights, df_test)

print(weights)
print(df_test)

err = computeError(df_test['y'].values , y_pred)
print(err)