import numpy as np
import math
import loadFittingDataP2 as ld

def main():
    x_data, y_data = ld.getData()
    print(poly_ml_sse(x_data, y_data, 5))
    
def poly_ml_sse(x_data, y_data, M):
    design_matrix = np.array([[x**m for m in range(M+1)] for x in x_data])
    return np.linalg.inv(design_matrix.T.dot(design_matrix)).dot(design_matrix.T).dot(y_data)

def sse():

def sse_d():

 

if __name__ == "__main__":
    main()
