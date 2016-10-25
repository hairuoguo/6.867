import sklearn.linear_model as lm
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import math
from cvxopt import matrix, solvers

def main():
    rbf_func = RBFSampler(gamma=1, random_state=1).fit_transform 

def pegasos(x_data, y_data, l, max_epochs):
    t = 0
    w = np.zeros(x_data.shape[1])
    w_0 = 0.
    num_epochs = 0
    while num_epochs < max_epochs:
        for n in y_data.size:
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
    alpha = np.zeros(x_data.shape[1])
    num_epochs = 0
    while num_epochs < max_epochs:
        for n in y_data.size:
            t += 1
            step_size = 1./(t*l)
            if y_data[n]*(alpha.reshape((-1, 1))*K) < 1.:
                alpha = (1 - step_size*l)*alpha + step_size*y_data[n]
            else:
                alpha = (1 - step_size*l)*alpha
        num_epochs += 1
    return alpha

if __name__ == "__main__":
    main()
