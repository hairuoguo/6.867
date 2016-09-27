import numpy as np
import math
import P2.loadFittingDataP2 as ld
import section1 as s1

def main():
    x_data, y_data = ld.getData()
    poly_funcs_constr = lambda n: lambda x:x**n
    cosine_funcs_constr = lambda n: lambda x: math.cos(n*math.pi*x)
    print(ml_sse(x_data, y_data, 5, poly_funcs_constr))
    poly_funcs, init_w = make_funcs_init_weights(5, poly_funcs_constr)
    print(make_sse_d(x_data, y_data, poly_funcs)(init_w))
    print(s1.finite_diff(init_w, 0.01, make_sse(x_data, y_data, poly_funcs)))
    batch_step_size = 10**-2
    batch_threshold = 10**-2
    batch_obj = make_sse(x_data, y_data, poly_funcs)
    batch_d = make_sse_d(x_data, y_data, poly_funcs)
    def sg_make_sse_d(x_data, y_data): return make_sse_d(x_data, y_data, poly_funcs)
    print(s1.batch_gd(x_data, y_data, batch_step_size, batch_threshold, batch_obj, batch_d, init_w)) 
    print(s1.stochastic_gd(x_data, y_data, batch_step_size, batch_threshold, batch_obj, sg_make_sse_d, init_w)) 
    print(ml_sse(x_data, y_data, 8, cosine_funcs_constr))

def make_funcs_init_weights(M, func_constr):
    funcs = [func_constr(n) for n in range(M)]
    init_w = np.random.normal(0, 0.1, M)
    return funcs, init_w
    
def ml_sse(x_data, y_data, M, func_constr):
    design_matrix = np.array([[func_constr(m)(x) for m in range(M+1)] for x in x_data])
    return np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y_data)

def make_sse(x_data, y_data, basis_functions):
    features = np.array([[function(x) for function in basis_functions] for x in x_data])
    return lambda w: 0.5*np.sum((y_data - w.dot(features.T))**2) 

def make_sse_d(x_data, y_data, basis_functions):
    if np.isscalar(x_data):
        features = np.array([basis_function(x_data) for basis_function in basis_functions] )
    else:
        features = np.array([[basis_function(x) for basis_function in basis_functions] for x in x_data])
    return lambda w: -1*np.sum(features*(y_data - w.dot(features.T)).reshape((-1, 1)), axis=0)
    
 

if __name__ == "__main__":
    main()
