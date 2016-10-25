import sklearn.linear_model as lm
import section1 as s1
import section2 as s2
import section3 as s3
from sklearn.kernel_approximation import RBFSampler
import numpy as np
import math
from cvxopt import matrix, solvers

def main():
    data = {}
    for n in xrange(10):
        data[n] = np.array(np.loadtxt('data/mnist_digit_' + str(n) + '.csv')).reshape((-1, 28, 28))
    1v2_data, 1v2_labels, 1v2_indices = create_dataset([1], [7], data)
    3v5_data, 3v5_labels, 3v5_indices = create_dataset([3], [5], data)
    4v9_data, 4v9_labels, 4v9_indices = create_dataset([4], [9], data)
    multi_data, multi_labels, multi_indices =  create_dataset([0, 2, 4, 6, 8], [1, 3, 5, 7, 9], data)
     
    
def create_dataset(pos_list, neg_list, data):
    train_data = np.array([])
    val_data = np.array([])
    test_data = np.array([])
    for n in pos_list:
        np.append(train_data, data[n][0:200, :, :])
        np.append(val_data, data[n][200:350, :, :])
        np.append(test_data, data[n][350:500, :, :])
    for m in neg_list:
        np.append(train_data, data[m][0:200, :, :])
        np.append(val_data, data[m][200:350, :, :])
        np.append(test_data, data[m][350:500, :, :])
    train_labels = np.append(np.ones(len(pos_list)*200), -1*np.ones(len(pos_list)*200))
    val_labels = np.append(np.ones(len(pos_list)*150), -1*np.ones(len(pos_list)*150))
    test_labels = np.append(np.ones(len(pos_list)*150), -1*np.ones(len(pos_list)*150))
    train_indices = np.random.shuffle(xrange(train_labels.size))
    val_indices = np.random.shuffle(xrange(val_labels.size))
    test_indices = np.random.shuffle(xrange(test_labels.size))
    data = {}
    data["train"] = train_data
    data["test"] = test_data
    data["val"] = val_data
    labels = {}
    labels["train"] = train_labels
    labels["val"] = val_labels
    labels["test"] = test_labels
    indices = {}
    indices["train"] = train_indices
    indices["test"] = test_indices
    indices["val"] = val_indices 
    return data, labels, indices 
    

if __name__ == "__main__":
    main()
