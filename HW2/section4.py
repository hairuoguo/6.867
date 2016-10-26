import sklearn.linear_model as lm
import section1 as s1
import section2 as s2
import section3 as s3
from sklearn.metrics import zero_one_loss
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import math
from cvxopt import matrix, solvers

def main():
    data = {}
    for n in xrange(10):
        data[n] = np.array(np.loadtxt('data/mnist_digit_' + str(n) + '.csv')).reshape((-1, 784))
    data1, labels1 = create_dataset([1], [7], data)
    data2, labels2 = create_dataset([3], [5], data)
    data3, labels3 = create_dataset([4], [9], data)
    multi_data, multi_labels = create_dataset([0, 2, 4, 6, 8], [1, 3, 5, 7, 9], data)
    datasets = [(data1, labels1), (data2, labels2), (data3, labels3), (multi_data, multi_labels)]    

    for dataset in datasets:
        x_data = dataset[0]['train']/255.
        y_data = dataset[1]['train'].reshape(-1, 1)
        x_test = dataset[0]['test']/255.
        y_test = dataset[1]['test'].reshape(-1, 1)
        '''
        w, w_0 = s3.pegasos(x_data, y_data, 0.001, 1000)
        '''
        q, p, A, b, I, G, h = s2.create_variables(x_data, y_data, 1.0)
        w = np.array(solvers.qp(p, q, G, h, A, b)['x'])
        results = (y_test*np.sign(w.dot(x_test.T)))
        print((results[results<0]).size)
        #print(y_data)
        '''
        l1 = lm.LogisticRegression('l1', fit_intercept=True, C=100)
        l2 = lm.LogisticRegression('l2', fit_intercept=True, C=100)
        l1.fit(x_data, y_data)
        l2.fit(x_data, y_data)
        l1_pred = l1.predict(x_test)
        l1_error = zero_one_loss(y_test, l1_pred, normalize=False)
        l2_pred = l2.predict(x_test)
        l2_error = zero_one_loss(y_test, l2_pred, normalize=False)
        '''
        #print(l1.coef_)
        #print(l2.coef_) 
        #print(l1_error)
        #print(l2_error)
        
    
    
        
        #q, p, A, b, I, G, h = create_variables(x_data, y_data, 0, rbf_func)
        #x = solvers.qp(p, q, G, h, A, b)['x']
     
    
def create_dataset(pos_list, neg_list, data):
    train_data = np.array([])
    val_data = np.array([])
    test_data = np.array([])
    for n in pos_list:
        if pos_list.index(n) == 0:
            train_data = data[n][0:200, :]
            val_data = data[n][200:350, :]
            test_data = data[n][350:500, :]
        else:
            train_data = np.append(train_data, data[n][0:200, :], axis=0)
            val_data = np.append(val_data, data[n][200:350, :], axis=0)
            test_data = np.append(test_data, data[n][350:500, :], axis=0)
    for m in neg_list:
        train_data = np.append(train_data, data[m][0:200, :], axis=0)
        val_data = np.append(val_data, data[m][200:350, :], axis=0)
        test_data = np.append(test_data, data[m][350:500, :], axis=0)
    train_labels = np.append(np.ones(len(pos_list)*200), -1*np.ones(len(pos_list)*200))
    val_labels = np.append(np.ones(len(pos_list)*150), -1*np.ones(len(pos_list)*150))
    test_labels = np.append(np.ones(len(pos_list)*150), -1*np.ones(len(pos_list)*150))
    train_indices = np.arange(train_labels.size)
    val_indices = np.arange(val_labels.size)
    test_indices = np.arange(test_labels.size)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    data = {}
    data["train"] = np.squeeze(train_data[train_indices, :])
    data["test"] = np.squeeze(test_data[test_indices, :])
    data["val"] = np.squeeze(val_data[val_indices, :])
    labels = {}
    labels["train"] = train_labels[train_indices].flatten()
    labels["val"] = val_labels[val_indices].flatten()
    labels["test"] = test_labels[test_indices].flatten()
    return data, labels 
    

if __name__ == "__main__":
    main()
