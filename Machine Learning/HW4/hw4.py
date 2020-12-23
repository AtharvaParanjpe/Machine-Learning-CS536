import numpy as np
import math

## Generating x, x_prime, epsilon
def generate_data(m,sigma_squared):
    x = np.random.uniform(100,102,m)
    x_prime = x -101
    epsilon = np.random.normal(0, math.sqrt(sigma_squared), m)
    return x, x_prime, epsilon

## Generating y
def generate_y(x, x_prime, epsilon, w, b):
    y = w*x + b + epsilon
    return y


## Function for computing weight and bias given x, y and number of samples
def computeWeightAndBias(x, y, m):
    sigmaXY = sum([a*b for a,b in zip(x,y)])
    sigmaX = sum(x)
    sigmaY = sum(y)
    sumSquaresX = sum([a**2 for a in x])
    sumX_2 = (sum(x))**2

    w_predicted = (m*sigmaXY - sigmaX*sigmaY)/(m*sumSquaresX - sumX_2)
    b_predicted = sum(y)/m - w_predicted*sum(x)/m

    return w_predicted, b_predicted


## Computing the different values
def analyze():
    m = 200
    w= 1
    b=5
    x = []
    w_prediction_1 = []
    b_prediction_1 = []
    w_prediction_2 = []
    b_prediction_2 = []
    sigma_squared = 0.1
    for i in range(1000):
        x, x_prime, epsilon = generate_data(m, sigma_squared)
        y = generate_y(x, x_prime, epsilon, w, b)
        w_1, b_1 = computeWeightAndBias(x, y, m)
        w_2, b_2 = computeWeightAndBias(x_prime, y, m)
        w_prediction_1.append(w_1)
        b_prediction_1.append(b_1)
        w_prediction_2.append(w_2)
        b_prediction_2.append(b_2)

    print("Wcap: ", sum(w_prediction_1)/len(w_prediction_1))
    print("Bcap: ", sum(b_prediction_1)/len(b_prediction_1))
    print()

    print("Wcap_prime: ", sum(w_prediction_2)/len(w_prediction_2))
    print("Bcap_prime: ", sum(b_prediction_2)/len(b_prediction_2))
    print()
    
    computedVarForW = np.var(w_prediction_1)
    computedVarForB = np.var(b_prediction_1)
    
    theoreticalVarForW = sigma_squared/(m*np.var(x))
    expXSquared = sum([a**2 for a in x])/len(x)
    theoreticalVarForB = sigma_squared*expXSquared/(m*np.var(x))

    print("Theoretical vs Actual For x")
    print("W: " ,computedVarForW, theoreticalVarForW)
    print("B: " , computedVarForB, theoreticalVarForB)
    print()
    
    print("---------------x------------------x---------------")
    
    print()
    print("Theoretical vs Actual For x prime")
    print()
    theoreticalVarForW = sigma_squared/(m*np.var(x_prime))
    expXSquared = sum([a**2 for a in x_prime])/len(x_prime)
    theoreticalVarForB = sigma_squared*expXSquared/(m*np.var(x_prime))
    

    computedVarForW = np.var(w_prediction_2)
    computedVarForB = np.var(b_prediction_2)
    
    print("W", computedVarForW, theoreticalVarForW)
    print("B", computedVarForB, theoreticalVarForB)

## Main call
analyze()