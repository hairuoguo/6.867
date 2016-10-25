import math
import numpy as np
import P4.lassoData as ld
import section2 as s2
import section1 as s1
from sklearn import linear_model as lm

def main():
    x_train, y_train = ld.lassoTrainData()
    x_val, y_val = ld.lassoValData()
    x_test, y_test = ld.lassoTestData()
    M = 13
    l = 0.1
    
    train_features = create_features(x_train, M)
    val_features = create_features(x_val, M)
    test_features = create_features(x_test, M)
    
    model = lm.Lasso(l)
    model.fit(train_features, y_train)
    print(x_train)
    print(model.coef_)
    print(model.intercept_)
    print(np.linalg.inv(train_features.T.dot(train_features)).dot(train_features.T).dot(y_train))

def create_features(x_data, M):
    features = np.array([[math.sin(0.4*math.pi*x*m) for m in range(1, M)] for x in x_data])
    features = np.hstack((x_data, features))
    return features 

if __name__ == "__main__":
    main()
