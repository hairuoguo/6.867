import sklearn.linear_model as lm
from sklearn.metrics import zero_one_loss
from plotBoundary import *
import matplotlib.pyplot as plt
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
    #print(gd_weights)
    '''
    fit_and_plot('l1', 0.00001)
    fit_and_plot('l2', 0.00001)
    fit_and_plot('l1', 1)
    fit_and_plot('l2', 1)
    '''
    '''
    get_error_from_data('data1', 'l1', 1.)
    get_error_from_data('data2', 'l2', 1.)
    get_error_from_data('data3', 'l1', 2.)
    get_error_from_data('data4', 'l1', 1.)
    '''
    fit_and_plot('l1', 1.)
    fit_and_plot('l1', 0.00001)
    fit_and_plot('l2', 1.)
    fit_and_plot('l2', 0.00001)

def get_error_from_data(data_name, reg_type, l):
    train = np.loadtxt('data/' + data_name + '_train.csv')
    x_train = np.array(train[:, 0:2])
    y_train = np.array(train[:, 2:3]).flatten()
    test = np.loadtxt('data/' + data_name + '_test.csv')
    x_test = np.array(test[:, 0:2])
    y_test = np.array(test[:, 2:3]).flatten()
    model = lm.LogisticRegression(reg_type, fit_intercept=True, C=1./l)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    error = zero_one_loss(y_test, pred, normalize=False)
    print(error)

def best_values():
    for n in xrange(4):
        data_name = 'data' + str(n+1)
        train = np.loadtxt('data/' + data_name + '_train.csv')
        x_train = np.array(train[:, 0:2])
        y_train = np.array(train[:, 2:3]).flatten()
        val = np.loadtxt('data/' + data_name + '_validate.csv')
        x_val = np.array(val[:, 0:2])
        y_val = np.array(val[:, 2:3]).flatten()
        for l in [1., 0.000001, 2, 2.5]:
            print(str(data_name) + ' ' + str(l)) 
            l1 = lm.LogisticRegression('l1', fit_intercept=True, C=1./l)
            l2 = lm.LogisticRegression('l2', fit_intercept=True, C=1./l)
            l1.fit(x_train, y_train)
            l2.fit(x_train, y_train)
            l1_pred = l1.predict(x_val)
            l1_error = zero_one_loss(y_val, l1_pred, normalize=False)
            l2_pred = l2.predict(x_val)
            l2_error = zero_one_loss(y_val, l2_pred, normalize=False)
            print(l1_error)
            print(l2_error)

def fit_and_plot(reg_type, l):
    print(reg_type + ' ' + str(l))
    for n in xrange(4):
        data_name = 'data' + str(n+1)
        train = np.loadtxt('data/' + data_name + '_train.csv')
        x_data = np.array(train[:, 0:2])
        y_data = np.array(train[:, 2:3]).flatten()
        model = lm.LogisticRegression(reg_type, fit_intercept=True, C=1./l)
        model.fit(x_data, y_data)
        print(np.linalg.norm(np.append(model.coef_, model.intercept_)))
        pred = model.predict(x_data)
        error = zero_one_loss(y_data, pred, normalize=False)
        X = x_data
        Y = y_data
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        h= .02
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        plt.subplot(2, 2, n + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
      # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.figure(1, figsize=(4, 3))

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        #plt.xlabel('Sepal length')
        #plt.ylabel('Sepal width')

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title('Data: ' + data_name + ' Model: ' + reg_type + ' lambda: ' + str(l) + ' Error: ' + str(error))
    #plt.show()




def make_nll_func(x_data, y_data, l):
    x_data = np.hstack((x_data, np.ones((x_data.shape[0], 1))))
    return lambda w: np.sum(np.log(1 + np.exp(-1*y_data*(w.dot(x_data.T))))) + l*np.linalg.norm(w[:-1])

def make_nll_d_func(x_data, y_data, l):
    x_data = np.hstack((x_data, np.ones((x_data.shape[0], 1))))
    return lambda w: np.sum((np.exp(-1*y_data*(w.dot(x_data.T)))*-1*y_data).reshape(-1, 1)*x_data/(1 + np.exp(-1*y_data*(w.dot(x_data.T)))).T.reshape(-1, 1) + l*np.append(2*w[:-1], 0.), axis=0)



if __name__ == "__main__":
    main()
