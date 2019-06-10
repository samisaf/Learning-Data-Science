import numpy as np
import scipy.io
import scipy.optimize 

def LogisticRegression(X, y, theta0=None, reg=0, optimizer='builtin', niter=100):
    """
    RETURNS theta/weights by fitting a logistic model using X (sample, features), and y labels.
    INPUT
    X: 2D array of m rows, and n cols (m, n) representing model samples and features
    y: 1D array of length m (m, ) representing labels
    theta: 1D array of length n (n, ) representing model weights
    reg: a number representing a regularization constant
    """
    # check if this is a binary or multi-class regression model
    if len(np.unique(y)) == 2: 
        return logistic_fit_binary(X, y, theta0, reg)
    else:
        return logistic_fit_multi(X, y, theta0, reg)

def logistic_fit_binary(X, y, theta0=None, reg=0, optimizer='builtin', niter=100):
    """
    fits a binary classification logistic model. This is an internal function called
    by LogisticRegression
    """
    # generates theta0 if not given
    m, n = X.shape
    if theta0 == None: theta0 = np.zeros(n + 1) # add 1 for the intercept
    # adds a column at the start for the intercept
    X = padwOnes(X)
    # calculate the weights
    # choose the optimizer
    optimizer = scipy.optimize.fmin_cg
    # define the cost and gradient as functions of theta
    # note that cost and gradient are functions since logistic_cost, and logistic gradient are partial functions
    cost = logistic_cost(X, y, reg)
    gradient = logistic_gradient(X, y, reg)
    # run the optimizer
    theta_optimum = optimizer(cost, theta0, fprime=gradient, maxiter=niter, disp=0)#,
    return theta_optimum

def sigmoid(z):
    """
    RETURN the sigmoid value(s) for a given number or array
    """
    return 1 / (1 + np.exp(-z))

def logistic_hypothesis(X, theta):
    """
    RETURNS the hypothesis/probability in a logistic model (number 0-1). This is a vectorized function.
    INPUT
    X: 2D array of m rows, and n cols (m, n) representing model samples and features
    theta: 1D array of length n (n, ) representing model weights
    """
    return sigmoid(X.dot(theta))

def logistic_cost(X, y, reg=0):
    """
    RETURNS the cost for a logistic model (number) at a given theta/weight. This is a vectorized partial function.
    INPUT
    X: 2D array of m rows, and n cols (m, n) representing model samples and features
    y: 1D array of length m (m, ) representing labels
    theta: 1D array of length n (n, ) representing model weights
    reg: a number representing a regularization constant
    """
    # (hypo==0) & (y==1)
    epsilon = 1e-100  
    m, n = X.shape
    def on(theta):
        hypo = logistic_hypothesis(X, theta)
        costs = -y * np.log(hypo + epsilon) - (1 - y) * np.log(1 - hypo + epsilon)
        #costs = np.log(hypo.where((hypo!=0) & (y==1))) - np.log(1 - hypo.where((hypo!=1) & (y==0)))
        #costs = sum(np.log(hypo[y==1 and hypo!=0])) + sum(np.log(1-hypo[y==0] and hypo!=1))
        #costs = -y * np.log(hypo) - (1 - y) * np.log(1 - hypo)
        penalties = 0.5 * reg * theta[1:]**2
        return (sum(costs) + sum(penalties)) / m
    return on

def logistic_gradient(X, y, reg=0):
    """
    RETURNS the gradient for a logistic model (number) at a given theta/weight. This is a vectorized partial function.
    INPUT
    X: 2D array of m rows, and n cols (m, n) representing model samples and features
    y: 1D array of length m (m, ) representing labels
    theta: 1D array of length n (n, ) representing model weights
    reg: a number representing a regularization constant
    """    
    def on(theta):
        m, n = X.shape
        hypo = logistic_hypothesis(X, theta)
        gradients = X.T.dot(hypo - y) / m
        penalties = np.append(0, reg * theta[1:]) / m
        return gradients + penalties
    return on

def padwOnes(X):
    """
    RETURNS the given array padded with a column of ones on the left side
    INPUT a two dimentional array X
    """
    m, n = X.shape
    ones = np.ones((m, 1))
    return np.append(ones, X, axis=1)

def test_binary():
    # load the data where X are the scores of two exams, and y is if the candidate was admitted to college
    data = np.genfromtxt('ex2data1.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    # run the model
    theta = LogisticRegression(X, y)
    # print theta to screen
    print('theta:')
    print(' %s \n' % theta)
    print('Expected theta (approx):')
    print(' -25.161 0.206 0.201\n')
    
    theta_expected = np.array([-25.161, 0.206, 0.201])
    threshold = 0.1
    assert sum(abs(theta -theta_expected)) < threshold

def test():
    test_binary()
    
if __name__=="__main__": test()