import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math


## pts inside the hemisphere
def z_data(k, m, epsilon):
    mu = 0
    sigma = 1
    data = []
    while(len(data)<m):
        z = np.random.normal(mu, sigma, k)
        modZ = LA.norm(z,2)
        z= z/modZ
        if(abs(z[-1])>=epsilon):
            data.append(z.tolist())
    return data

## discard pts < epsilon and generate targets for remaining data
def generate_targets(data):
    data_rows = []
    for x in data:
        row = []
        if(x[-1]>=0):
            row+=x
            row+=[1]
        else:
            row+=x
            row+=[-1]
        data_rows.append(row)
    data_rows.append(row)
    columns = []
    for i in range(len(data[0])):
        columns+=['x' + str(i+1)]
    columns.append('y')
    
    df = pd.DataFrame(data_rows, columns=columns)
    return df

## fit perceptron
def perceptron_model(train_data):
    flag = False
    targets = train_data['y'].values
    train_data = train_data.drop(["y"], axis=1)
    weight_vector = np.zeros(len(train_data.values[0]))
    bias = 0
    steps = 0
    error = 0
    while(not flag):
        flag = True
        for i in range(len(train_data["x1"].values)):
            x = np.array(train_data.values[i])
            out = np.dot(weight_vector,np.transpose(x)) + bias

            y = 1 if out>=0 else -1
            if y!=targets[i]:
                flag = False
                weight_vector = weight_vector + targets[i]*x
                bias = bias + targets[i]
                steps+=1
    return weight_vector, bias, steps


## Part 1 m 0-2500
m = []
steps = []
for i in range(0,2500,10):
    m.append(i)
    count = 0
    for j in range(10):
        z = z_data(5, i+1, 0.1)
        data = generate_targets(z)
        w,b,s =  perceptron_model(data)
        count +=  s
    count = count *1.0/10
    steps.append(count)
    if(i%200==0):
        print(i)

plt.plot(m, steps)
plt.xlabel("M")
plt.ylabel("Steps")
plt.show()

##Part 2 k 0-100

k = []
steps = []
for i in range(0,500):
    k.append(i)
    count = 0
    for j in range(20):
        z = z_data(i+1, 100, 0.05)
        data = generate_targets(z)
        w,b,s =  perceptron_model(data)
        count +=  s
    count = count *1.0/20
    steps.append(count)
    # print(i)

plt.plot(k, steps)
plt.xlabel("k")
plt.ylabel("Steps")
plt.show()

# Part 3 ## epsilon 0.02-098

ep = []
steps = []
for i in range(2,98):
    ep.append(i*1.0/100)
    count = 0
    for j in range(20):
        z = z_data(5, 100, i*1.0/100)
        data = generate_targets(z)
        w,b,s =  perceptron_model(data)
        count +=  s
    count = count *1.0/20
    data = generate_targets(z)
    steps.append(count)
    # print(i)

plt.plot(ep, steps)
plt.xlabel("epsilon")
plt.ylabel("Steps")
plt.show()


############################## BONUS QUESTION ################################
def calcParameter(weights, bias):
    true_weights = np.zeros(len(weights))
    true_weights[-1] = 1
    norm = math.sqrt(LA.norm(weights,2)**2 + bias**2)
    weights = weights/norm
    bias = bias/norm
    true_bias = 0
    value = LA.norm((true_weights-weights),2)**2 + (true_bias - bias)**2
    return value



## Part 1 ##

m = []
specialParameter = []

for i in range(0,250):
    m.append(i)
    weights = 0
    bias = 0
    count = 0
    for j in range(10):
        z = z_data(5, i+1, 0.1)
        data = generate_targets(z)
        w,b,s =  perceptron_model(data)
        count +=  s
        weights+= w
        bias+= b
    count = count *1.0/10
    weights = weights *1.0/10
    bias = bias *1.0/10
    specialParameter.append(calcParameter(weights, bias))
    if(i%200==0):
        print(i)

plt.plot(m, specialParameter)
plt.xlabel("M")
plt.ylabel("||w* - w||^2 + (b*-b)^2")
plt.show()



## Part 2 ##
k = []
specialParameter = []

for i in range(9,500,10):
    k.append(i)
    weights = 0
    bias = 0
    count = 0
    for j in range(100):
        z = z_data( i+1, 100, 0.05)
        data = generate_targets(z)
        w,b,s =  perceptron_model(data)
        count +=  s
        weights+= w
        bias+= b
    count = count *1.0/100
    weights = weights *1.0/100
    bias = bias *1.0/100
    specialParameter.append(calcParameter(weights, bias))
    if(i%200==0):
        print(i)

plt.plot(k, specialParameter)
plt.xlabel("K")
plt.ylabel("||w* - w||^2 + (b*-b)^2")
plt.show()



## Part 3 ##

ep = []
specialParameter = []

for i in range(2,98,2):
    ep.append(i*1.0/100)
    weights = 0
    bias = 0
    count = 0
    for j in range(100):
        z = z_data(5, 100, i*1.0/100)
        data = generate_targets(z)
        w,b,s =  perceptron_model(data)
        count +=  s
        weights+= w
        bias+= b
    count = count *1.0/100
    weights = weights *1.0/100
    bias = bias *1.0/100
    specialParameter.append(calcParameter(weights, bias))

plt.plot(ep, specialParameter)
plt.xlabel("epsilon")
plt.ylabel("||w* - w||^2 + (b*-b)^2")
plt.show()