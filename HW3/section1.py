import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

    
class FCLayer:
    
    def __init__(self, in_layer, num_nodes, activation_func, gamma, gamma_b, delta):
        self.num_nodes = num_nodes
        self.in_layer = in_layer
        self.w = np.random.normal(0.0, 1./math.sqrt(self.in_layer.num_nodes), (self.num_nodes, self.in_layer.num_nodes)) 
        self.b = np.ones(self.num_nodes) 
        self.gamma = gamma
        self.gamma_b = gamma_b
        self.activation_func = activation_func
        self.delta = delta
        self.recent_x = None
        self.recent_z = None
            

    def compute(self, x):
        #recursively compute by calling previous layer's compute
        self.recent_x = self.in_layer.compute(x)
        self.recent_z = self.recent_x.dot(self.w.T) + self.b
        #print(self.recent_z)
        return self.activation_func.func(self.recent_z)
    
    def backprop(self, prev_grad, prev_w, step):
        gradient = ((self.activation_func.gradient_func(self.recent_z)*np.identity(self.num_nodes)).dot(prev_w.T)).dot(prev_grad)
        self.in_layer.backprop(gradient, self.w, step)
        self.w = self.w - math.exp(1 - self.delta*step)*self.gamma*(self.recent_x.reshape((1, self.recent_x.size))*np.tile(gradient.reshape((-1, 1)), (1, self.recent_x.size)))
        self.b = self.b - math.exp(1 - self.delta*step)*self.gamma_b*gradient

class InputLayer:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def compute(self, x):
        assert x.size == self.num_nodes
        return x

    def backprop(self, prev_grad, w, step):
        return

class Function:
    #elementwise function 
    def __init__(self, func, gradient_func):
        self.func = func
        self.gradient_func = gradient_func
    
     

def main():

    relu = lambda x: np.maximum(x, 0.)
    relu_grad = lambda x: np.array([1. if n>0 else 0. for n in x])
    relu_func = Function(relu, relu_grad)
    softmax = lambda x: np.exp(x)/np.sum(np.exp(x))
    softmax_grad = lambda x: np.array([[n*(1-n) if n == m else -1*n*m for n in softmax(x)] for m in softmax(x)])
    softmax_func = Function(softmax, softmax_grad)
    cross_entropy_loss = lambda x, y: -1*np.log(x).dot(y.T)
    cross_entropy_grad = lambda x, y: -1.*(1./x)*y
    #cross_entropy_grad = lambda x, y: x - y
    cross_entropy_func = Function(cross_entropy_loss, cross_entropy_grad)

    data_hidden_1 = create_single_hidden(20, 10**-3, 10**-8, 10**-15, relu_func, softmax_func)
    #data1_hidden_1_large
    data_hidden_2 = create_double_hidden(100, 500, 10**-7, 10**-10, 10**-15, relu_func, softmax_func)
    #data1_hidden_2_large

    
    data = np.loadtxt('data/data4_train.csv')
    data_val = np.loadtxt('data/data4_validate.csv')
    data_test = np.loadtxt('data/data4_test.csv')
    val_x = data_val[:, 0:2]
    val_y = data_val[:, 2]
    test_x = data_val[:, 0:2]
    test_y = data_val[:, 2]
    data_x = data[:, 0:2]
    data_x[:, 0] = (data_x[:, 0] - np.mean(data_x[:, 0]))/np.ptp(data_x[:, 0])
    data_x[:, 1] = (data_x[:, 1] - np.mean(data_x[:, 1]))/np.ptp(data_x[:, 1])
    data_y = data[:, 2]
    '''
    for n in xrange(10):
        digit_pixels = np.array(np.loadtxt('data/mnist_digit_' + str(n) + '.csv')).reshape((-1, 784))/255.
        if n == 0:
            data_x = digit_pixels[0:200, :]
            data_y = np.array([n]*200)
            val_x = digit_pixels[200:350, :]
            val_y = np.array([n]*150)
            test_x = digit_pixels[350:500, :]
            test_y = np.array([n]*150)
        else:
            data_x = np.append(data_x, digit_pixels[0:200, :], axis=0)
            data_y = np.append(data_y, np.array([n]*200), axis=0)
            val_x = np.append(val_x, digit_pixels[200:350, :], axis=0)
            val_y = np.append(val_y, np.array([n]*150), axis=0)
            test_x = np.append(test_x, digit_pixels[350:500, :], axis=0)
            test_y = np.append(test_y, np.array([n]*150), axis=0)
    
    ''' 
    errors, val_errors = sgd(data_x, data_y, val_x, val_y, 1000000, data_hidden_1, cross_entropy_func)
    test_network(data_x, data_y, data_hidden_1)
    test_network(test_x, test_y, data_hidden_1)
    plt.plot(range(len(errors)), errors, 'b-', range(len(errors)), val_errors, 'g-', linewidth=2.0)
    plt.title('Data4, 1 big layer')
    plt.xlabel('10^3 iterations')
    plt.ylabel('Cross-entropy loss')
    green = mpatches.Patch(color='green', label='Validation error')
    blue = mpatches.Patch(color='blue', label='Training error')
    plt.legend(handles=[green, blue])
    plt.show()
    #sgd(data_x, data_y, val_x, val_y, 1000000, n_2_output, cross_entropy_func)

def test_network(data_x, data_y, network):
    val_error = 0.
    for n in xrange(data_y.size):
        x_val_sample = data_x[n]
        y_val_sample = one_hot_ones(data_y[n])
        output_val = network.compute(x_val_sample)
        error = min(abs(np.argmax(output_val) - data_y[n]), 1) 
        val_error += error 
    print("Error is: " + str(val_error/data_y.size))

def create_single_hidden(size, gamma, gamma_b, delta, relu_func, softmax_func):
    input_layer = InputLayer(2)
    n_1_fc_1 = FCLayer(input_layer, size, relu_func, gamma, gamma_b, delta)
    n_1_output = FCLayer(n_1_fc_1, 2, softmax_func, 10**-6, 10**-8, 10**-15)
    return n_1_output 

def create_double_hidden(size_1, size_2, gamma, gamma_b, delta, relu_func, softmax_func):
    input_layer = InputLayer(2)
    n_2_fc_1 = FCLayer(input_layer, size_1, relu_func, gamma, gamma_b, delta)
    n_2_fc_2 =  FCLayer(n_2_fc_1, size_2, relu_func, gamma, gamma_b, delta)
    n_2_output = FCLayer(n_2_fc_2, 2, softmax_func, 10**-6, 10**-8, 10**-15)
    return n_2_output

def one_hot_ones(index):
    y = np.zeros(2)
    if index == 1:
        y[0] = 1
    else:
        y[1] = 1
    return y

def one_hot_zero_one(size, index):
    y = np.zeros(size)
    y[index] = 1
    return y 

def sgd(data_x, data_y, x_val, y_val, num_iters, output_layer, loss_func):
    sum_error = 0
    prev_val_error = 1000000.
    indices = range(data_y.shape[0])
    val_errors = []
    errors = []
    for step in xrange(num_iters):
        if step % data_y.shape[0] == 0:
            np.random.shuffle(indices)
        index = indices[step % data_y.shape[0]]
        x_sample = data_x[index, :]
        #y_sample[data_y[index]] = 1
        y_sample = one_hot_ones(data_y[index])
        output = output_layer.compute(x_sample)
        #print(output)
        #print(y_sample)
        error = loss_func.func(output, y_sample)
        output_layer.backprop(loss_func.gradient_func(output, y_sample), np.identity(output.size), step)
        if math.isnan(error):
            return
        sum_error += error
        if step % 1000 == 0 and step != 0:
            print("Step: " + str(step))
            print(sum_error/1000)
            val_error = 0.
            for n in xrange(y_val.size):
                x_val_sample = x_val[n]
                y_val_sample = one_hot_ones(y_val[n])
                output_val = output_layer.compute(x_val_sample)
                val_error += loss_func.func(output_val, y_val_sample)
            val_error = val_error/y_val.size
            val_errors.append(val_error)
            errors.append(sum_error/1000)
            sum_error = 0.
            print("Val error is: " + str(val_error))
            if val_error > prev_val_error: return errors, val_errors
            prev_val_error = val_error
    print(step)
    return errors, val_errors 


if __name__ == "__main__":
    main()
