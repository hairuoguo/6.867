import sklearn.linear_model as lm
import numpy as np
import math
import sgd as gd

def main():

train = loadtxt('data/data1_train.csv')
train_x = train[:, 0:2]
train_y = train[:, 2:3]
gd_step_size = 
gd_threshold = 
gd_guess = 
nll = make_nll_func(train_x, train_y, 0) 
nll_d = make_nll_d_func(train_x, train_y, 0)
gd.basic_gd(gd_guess, gd_step_size, gd_threshold, nll, nll_d)

l = 
l1_model = lm.LogisticRegression('l1', C=1./l)
l2_model = lm.LogisticRegression('l2', C=1./l)
l1_model.fit(train_x, train_y)
l2_model.fit(train_x, train_y)



def make_nll_func(x_data, y_data, l):
    x_data = np.hstack((x_data, np.ones(x_data.shape[0])))
    return lambda w: np.sum(np.log(1 + np.exp(-1*y_data*(x_data.dot(w))))) + l*np.linalg.norm(w[:-1])

def make_nll_d_func(x_data, y_data, l):
    x_data = np.hstack((x_data, np.ones(x_data.shape[0])))
    return lambda w: np.exp(-1*y_data*(x_data.dot(w))*-1*y_data.T*w/(1 + np.exp(-1*y_data*(x_data.dot(w)))).T



if __name__ == "__main__":
    main()
