import sklearn.linear_model as lm
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import math
from cvxopt import matrix, solvers

def main():
    x_data = np.array([[2., 2.],
    [2., 3.],
    [0., -1.],
    [-3., -2.]
    ])
    y_data = np.array([1., 1., -1., -1.]).reshape(-1, 1)
    q, p, A, b, I, G, h = create_variables(x_data, y_data, 1.0)
    x = solvers.qp(p, q, G, h, A, b)['x']
    print(x)
    rbf_func = RBFSampler(gamma=1, random_state=1).fit_transform
    q, p, A, b, I, G, h = create_variables(x_data, y_data, 1.0, rbf_func)
    x = solvers.qp(p, q, G, h, A, b)['x']
    print(x)
    
    
    

def create_variables(x_data, y_data, c, map_func=lambda x:x):
    q = -1*matrix(np.ones((y_data.size, 1)))
    x_data = map_func(x_data)
    p = matrix(np.multiply(y_data, (np.multiply(y_data.T, (x_data.dot(x_data.T))))))
    A = matrix(y_data.T)
    b = matrix(0.)
    I = matrix(np.identity(y_data.size))
    G = matrix(np.vstack((I, -1*I)))
    h = matrix(np.vstack((c*np.ones((y_data.size, 1)), np.zeros((y_data.size, 1)))))
    return q, p, A, b, I, G, h
    

if __name__ == "__main__":
    main()
