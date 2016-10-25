import sklearn.linear_model as lm
import numpy as np
import math
import sgd as gd

def main():

    train = np.loadtxt('data/data1_train.csv')
    train_x = np.array(train[:, 0:2])
    train_y = np.array(train[:, 2:3]).flatten()
    gd_step_size = 10**-3 
    gd_threshold = 10**-10
    gd_guess = np.zeros(train_x.shape[1] + 1)
    nll = make_nll_func(train_x, train_y, 1) 
    nll_d = make_nll_d_func(train_x, train_y, 1)
    gd_weights = gd.basic_gd(gd_guess, gd_step_size, gd_threshold, nll, nll_d)
    print(gd_weights)

    l = 1
    l1_model = lm.LogisticRegression('l1', fit_intercept=True, C=1./l)
    l2_model = lm.LogisticRegression('l2', fit_intercept=True, C=1./l)
    l1_model.fit(train_x, train_y)
    l2_model.fit(train_x, train_y)
    print(l1_model.coef_)
    print(l2_model.coef_)



def make_nll_func(x_data, y_data, l):
    x_data = np.hstack((x_data, np.ones((x_data.shape[0], 1))))
    return lambda w: np.sum(np.log(1 + np.exp(-1*y_data*(w.dot(x_data.T))))) + l*np.linalg.norm(w[:-1])

def make_nll_d_func(x_data, y_data, l):
    x_data = np.hstack((x_data, np.ones((x_data.shape[0], 1))))
    return lambda w: np.sum((np.exp(-1*y_data*(w.dot(x_data.T)))*-1*y_data).reshape(-1, 1)*x_data/(1 + np.exp(-1*y_data*(w.dot(x_data.T)))).T.reshape(-1, 1) + l*np.append(2*w[:-1], 0.), axis=0)



if __name__ == "__main__":
    main()
