import math
import numpy as np

        
class OutputLayer:

    def __init__(self, in_layer, output_func):
        self.in_layer = in_layer
        self.output_func = output_func
        self.recent_x = None

    def compute(self, x):
        self.recent_x = x
        return self.output_func.func(self.in_layer.compute(x))

    def backprop(self, loss_grad):
        gradient = self.output_func.gradient_func(self.recent_x).dot(loss_grad)
        weights = np.ones(self.recent_x.size)
        self.in_layer.backprop(gradient, weights)
        

class FCLayer:
    
    def __init__(self, in_layer, num_nodes, activation_func, gamma, gamma_b):
        self.num_nodes = num_nodes
        self.in_layer = in_layer
        self.w = np.repeat(np.random.normal(0.0, 1./self.in_layer.num_nodes, self.in_layer.num_nodes), num_nodes, axis=0) 
        self.b = np.repeat(np.array([0.]*self.in_layer.num_nodes), num_nodes, axis=0)
        self.gamma = gamma
        self.gamma_b = gamma_b
        self.recent_x = None
        self.recent_z = None
            

    def compute(self, x):
        #recursively compute by calling previous layer's compute
        self.recent_x = x
        self.recent_z = self.in_layer.compute(x)
        return activation_func.func(self.recent_z.dot(w) + b)
    
    def backprop(self, prev_grad, prev_w):
        gradient = activation_func.gradient_func(self.recent_x)*np.identity(num_nodes).dot(prev_w).dot(prev_grad)
        self.in_layer.backprop(gradient, self.w)
        self.w = self.w - gamma*self.recent_x.dot(gradient.T)*self.w
        self.b = self.b - gamma_b*gradient*self.b

class InputLayer:

    def __init__(self, value):
        self.input_array = input_array
        self.num_nodes = input_array.size

    def compute(self):
        return self.input_array

    def backprop(self, prev_grad):
        return

class Function:
    #elementwise function 
    def __init__(self, func, gradient_func):
        self.func = func
        self.gradient_func = gradient_func
    
        

def main():

    relu = lambda x: np.maximum(x, 0)
    relu_grad = lambda x: np.array([1 if n >= 0 else 0 for n in x])
    softmax = lambda x: 
    sofmax_grad = lambda x:
    cross_entropy_loss = lambda x, y: 
    cross_entropy_grad = lambda x, y:


def batch_gd(data_x, data_y, step_size, threshold, obj_func, d_func, init_theta):
    theta = init_theta
    prev_error = obj_func(theta)
    convergence = False
    step = 0
    while not convergence:
        gradient = d_func(theta)
        theta = -1*gradient*step_size + theta
        error = obj_func(theta)
        if math.isnan(error):
            return
        convergence = abs(error - prev_error) < threshold
        prev_error = error
        step += 1
        #print(error)
    print(step)
    print("Final avg error: " + str(error))
    return theta  


if __name__ == "__main__":
    main()
