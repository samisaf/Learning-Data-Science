#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools

# Part 2: Plotting
data = np.genfromtxt ('ex1data1.txt', delimiter=",")
X = np.matrix(data[:, 0]).T
y = np.matrix(data[:, 1]).T
m = len(y)
plt.scatter(X, y, alpha=0.7)
plt.show()
ones = np.ones((m, 1))
X = np.hstack((ones, X)) # Add a column of ones to x

# Part 3: Cost and Gradient descent
def derive(f):
    def dfdx(x, step):
        dy = f(x + step) - f(x)
        dx = step
        return dy/dx
    return dfdx

def computeCost(X, y, theta):
    m = len(y)
    costs = np.power((X*theta - y), 2)
    J = (sum(costs)) / (2*m)
    return J

partialComputeCost = functools.partial(computeCost, X, y)
dcostdtheta = derive(partialComputeCost)

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    #print("theta0", theta)
    for i in range(num_iters):
        J_history[i] = computeCost(X, y, theta)
        #print("derviation", derive(partialCost, i).T)
        for j in range(len(theta)):
            temp = np.matrix(np.zeros(len(theta))).T
            temp[j] = alpha
            #print(temp)
            theta[j] = theta[j] - alpha * dcostdtheta(theta, temp)[j]
        #print(J_history[i])
        print(theta)
        
    print("Result, theta = ", theta, "with cost = ", J_history[-1])
    return theta, J_history

theta = np.matrix('1 ; 1')
iterations, alpha = 10, 0.001 # Some gradient descent settings

theta, cost = gradientDescent(X, y, theta, alpha, iterations);