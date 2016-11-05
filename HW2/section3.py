import sklearn.linear_model as lm
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import math
from cvxopt import matrix, solvers

def main():
    #print(pegasos(x_train, y_train, 1, 1000))
    #pegasos_kernel(x_train, y_train, 1, 1000, rbf_func)
    margins = []
    #print(x_train.shape)
    for n in xrange(1, 5):
        train = np.loadtxt('data/data' + str(n) + '_train.csv')
        test = np.loadtxt('data/data' + str(n) + '_validate.csv')
        x_train = train[:, 0:2].copy()
        y_train = train[:, 2:3].copy().flatten()
        x_test = test[:, 0:2].copy()
        y_test = test[:, 2:3].copy().flatten()
        print(y_test.shape)
        rbf_func = RBFSampler(gamma=1, random_state=1).fit_transform
        alpha = pegasos_kernel(x_train, y_train, 0.02, 1000, rbf_func)[:y_test.size]
        x_data = x_test
        x_data = rbf_func(x_test)
        K = x_test.dot(x_test.T)
        results = y_test*np.sign(alpha.dot(K))
        print(sum(results[results>0]))

def pegasos(x_data, y_data, l, max_epochs):
    t = 0
    w = np.zeros(x_data.shape[1])
    w_0 = 0.
    num_epochs = 0
    while num_epochs < max_epochs:
        for n in xrange(y_data.size):
            t += 1
            step_size = 1./(t*l)
            if y_data[n]*(w.dot(x_data[n]) + w_0) < 1.:
                w = (1 - step_size*l)*w + step_size*y_data[n]*x_data[n]
                w_0 = w_0 + step_size*y_data[n]
            else:
                w = (1 - step_size*l)*w
        num_epochs += 1
    return w, w_0


def pegasos_kernel(x_data, y_data, l, max_epochs, map_func=lambda x:x):
    t = 0
    x_data = map_func(x_data)
    K = x_data.dot(x_data.T)
    alpha = np.zeros(y_data.size)
    num_epochs = 0
    while num_epochs < max_epochs:
        for n in xrange(y_data.size):
            t += 1
            step_size = 1./(t*l)
            if y_data[n]*np.sum(alpha*K[:, n]) < 1.:
                alpha[n] = (1 - step_size*l)*alpha[n] + step_size*y_data[n]
            else:
                alpha[n] = (1 - step_size*l)*alpha[n]
        num_epochs += 1
    return alpha

if __name__ == "__main__":
    main()
