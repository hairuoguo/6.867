import math
import numpy as np
import section2 as s2
import P2.loadFittingDataP2 as ld
import P3.regressData as rd

def main():
    x_data, y_data = ld.getData()
    poly_funcs_constr = lambda n: lambda x:x**n
    M = 8 
    l = 0.5
    x_A, y_A = rd.regressAData()
    x_A = x_A.flatten()
    y_A = y_A.flatten()
    x_B, y_B = rd.regressBData()
    x_B = x_B.flatten()
    y_B = y_B.flatten()
    x_val, y_val = rd.validateData()
    x_val = x_val.flatten()
    y_val = y_val.flatten()
    optimal_weights = calculate_optimal_weights(x_data, y_data, l, M, poly_funcs_constr)
    optimal_weights_A = calculate_optimal_weights(x_A, y_A, l, M, poly_funcs_constr)
    optimal_weights_B = calculate_optimal_weights(x_B, y_B, l, M, poly_funcs_constr)
    optimal_weights_val = calculate_optimal_weights(x_val, y_val, l, M, poly_funcs_constr)
    print(optimal_weights)
    print(make_ridge_regression(x_data, y_data, l, M, poly_funcs_constr)(optimal_weights))
    print(make_ridge_regression(x_B, y_B, l, M, poly_funcs_constr)(optimal_weights_A))
    print(make_ridge_regression(x_A, y_A, l, M, poly_funcs_constr)(optimal_weights_B))
    print(make_ridge_regression(x_val, y_val, l, M, poly_funcs_constr)(optimal_weights_A))
    print(make_ridge_regression(x_val, y_val, l, M, poly_funcs_constr)(optimal_weights_B))

def make_ridge_regression(x_data, y_data, l, M, func_constr):
    funcs, _ = s2.make_funcs_init_weights(M, func_constr)
    features = np.array([[func(x) for func in funcs] for x in x_data])
    return lambda w: np.sum(0.5*(w.dot(features.T) - y_data)**2 + l/2*np.linalg.norm(w)**2)

def calculate_optimal_weights(x_data, y_data, l, M, func_constr):
    design_matrix = np.array([[func_constr(m)(x) for m in range(M)] for x in x_data])
    return np.linalg.inv(l*np.identity(design_matrix.shape[1]) + design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y_data)

if __name__ == "__main__":
    main()
