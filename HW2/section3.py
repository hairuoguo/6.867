import sklearn.linear_model as lm
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import math
from cvxopt import matrix, solvers

def main():
    train = np.loadtxt('data/data3_train.csv')
    x_train = train[:, 0:2].copy()
    y_train = train[:, 2:3].copy().flatten()
    #print(pegasos(x_train, y_train, 1, 1000))
    #pegasos_kernel(x_train, y_train, 1, 1000, rbf_func)
    margins = []
    #print(x_train.shape)
    for n in xrange(-2, 3):
        rbf_func = RBFSampler(gamma=2**(n), random_state=1).fit_transform
        alpha = pegasos_kernel(x_train, y_train, 0.02, 1000, rbf_func)
        print(n)
        print(1/np.linalg.norm(alpha))
        print(np.count_nonzero(alpha))
    

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
