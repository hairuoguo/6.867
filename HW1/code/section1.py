import numpy as np
import matplotlib.pyplot as plt
import math
import P1.loadFittingDataP1 as ld
import P1.loadParametersP1 as lp

def main():
    gaussMean, gaussCov, quadBowlA, quadBowlb = lp.getData()
    gauss_initial_guess = np.array([15., 15.])
    gauss_step_size = 10**6
    gauss_threshold = 10**-12
    quadbowl_initial_guess = np.array([40., 40.])
    quadbowl_step_size = 10**-3
    quadbowl_threshold = 10**-6
    neg_gauss = make_neg_gauss(gaussMean, gaussCov)
    quadbowl = make_quadbowl(quadBowlA, quadBowlb)
    neg_gauss_d = make_neg_gauss_d(gaussMean, gaussCov) 
    quadbowl_d = make_quadbowl_d(quadBowlA, quadBowlb)
    gd_x, gd_y = ld.getData()
    batch_threshold = 10**-3
    batch_step_size = 10**-5
    theta = np.random.normal(0, 0.1, gd_x.shape[1])
    tau = 10**10
    kappa = 0.5
    
    gauss_points = [gauss_initial_guess, np.array([10., 10.]), np.array([-10., -10.])]
    quad_points = [quadbowl_initial_guess, np.array([80./3, 80./3]), np.array([-20., -20.])]
    '''
    for x in [1, -1, -5]:
        print("Gauss" + str(x))
        for point in gauss_points:
            print(finite_diff(point, 10**x, neg_gauss))
        print("Quadbowl" + str(x))
        for point in quad_points:
            print(finite_diff(point, 10**x, quadbowl))
    '''
    '''
    dist_array = []
    for x in xrange(-30, 30): 
        gauss_initial_guess = np.array([x, x])
        result = basic_gd(gauss_initial_guess, gauss_step_size, gauss_threshold, neg_gauss, neg_gauss_d)
        dist_array.append(abs(neg_gauss([10., 10.]) - neg_gauss(result)))
    print(dist_array)
    plt.plot(-1*np.array(range(-30, 30)), dist_array)
    plt.xlabel('Initial guess of (x, x)')
    plt.ylabel('Error from true minimum')
    plt.title('Initial guess versus error for Gaussian')
    plt.show()
  
    print(basic_gd(quadbowl_initial_guess, quadbowl_step_size, quadbowl_threshold, quadbowl, quadbowl_d))
    
    dist_array = []
    for x in np.arange(23, 30, 0.2): 
        quadbowl_initial_guess = np.array([x, x])
        result = basic_gd(quadbowl_initial_guess, quadbowl_step_size, quadbowl_threshold, quadbowl, quadbowl_d)
        print(x)
        dist_array.append(abs(quadbowl(np.array([80./3, 80./3])) - quadbowl(result)))
    print(dist_array)
    plt.plot(np.array(np.arange(23, 30, 0.2)), dist_array)
    plt.xlabel('Initial guess of (x, x)')
    plt.ylabel('Error from true minimum')
    plt.title('Initial guess versus error for Quadbowl')
    plt.show()
  
    dist_array = []
    for x in xrange(-6, 0): 
        quadbowl_step_size = 10**x
        result = basic_gd(quadbowl_initial_guess, quadbowl_step_size, quadbowl_threshold, quadbowl, quadbowl_d)
        print(x)
        dist_array.append(abs(quadbowl(np.array([80./3, 80./3])) - quadbowl(result)))
    print(dist_array)
    plt.plot(np.array(range(-6, 0)), dist_array)
    plt.xlabel('Log base 10 of step size')
    plt.ylabel('Error from true minimum')
    plt.title('Step size versus error for Quadbowl')
    plt.show()
    '''


    #print(basic_gd(quadbowl_initial_guess, quadbowl_step_size, quadbowl_threshold, quadbowl, quadbowl_d))
    #print(finite_diff(gauss_initial_guess, 10**-5, neg_gauss)) 
    #print(finite_diff(quadbowl_initial_guess, 10**-1, quadbowl))
    avg_least_sq = make_least_sq(gd_x, gd_y)
    avg_least_sq_d = make_least_sq_d(gd_x, gd_y) 
    print(batch_gd(gd_x, gd_y, batch_step_size, batch_threshold,  avg_least_sq, avg_least_sq_d, theta))
    ideal_theta = np.linalg.inv(gd_x.T.dot(gd_x)).dot(gd_x.T).dot(gd_y)
    #print(ideal_theta)
    print(avg_least_sq(ideal_theta))
    print(stochastic_gd(gd_x, gd_y, tau, kappa, batch_threshold,  avg_least_sq, make_least_sq_d, theta))
    

def make_quadbowl(quadBowlA, quadBowlb):
    A = quadBowlA
    b = quadBowlb
    return lambda x: 0.5*x.T.dot(A).dot(x) - x.dot(b)

def make_neg_gauss(gaussMean, gaussCov):
    Sigma = gaussCov
    u = gaussMean
    return lambda x: -1*1/(math.sqrt((2*math.pi)**len(x)*np.linalg.det(Sigma)))*math.exp(-1./2*(x - u).dot(np.linalg.inv(Sigma)).dot(x - u))

def make_neg_gauss_d(gaussMean, gaussCov):
    Sigma = gaussCov
    u = gaussMean
    return lambda x: -1*-1*1/(math.sqrt((2*math.pi)**len(x)*np.linalg.det(Sigma)))*math.exp(-1./2*(x - u).dot(np.linalg.inv(Sigma)).dot(x - u))*(np.linalg.inv(Sigma)).dot(x-u)

def make_quadbowl_d(quadBowlA, quadBowlb):
    return lambda x: quadBowlA.dot(x) - quadBowlb

def finite_diff(x, delta, function):
    all_deltas = []
    for n in xrange(len(x)):
        temp_x_1 = np.copy(x)
        temp_x_1[n] = temp_x_1[n] + 0.5*delta
        temp_x_2 = np.copy(x)
        temp_x_2[n] = temp_x_2[n] - 0.5*delta
        all_deltas.append(function(temp_x_1) - function(temp_x_2))
    return np.array(all_deltas)/delta

def make_least_sq(batch_x, batch_y):
    return lambda theta: np.sum((batch_x.dot(theta) - batch_y)**2)    

def make_least_sq_d(batch_x, batch_y):
    return lambda theta: 2*batch_x.T.dot(batch_x.dot(theta) - batch_y)
     

def basic_gd(initial_guess, step_size, threshold, obj_func, d_func):
    x = initial_guess
    prev_y = obj_func(x) 
    convergence = False
    g_norms = []
    while not convergence:
        gradient = d_func(x)
        g_norms.append(np.linalg.norm(gradient))
        x = -1*gradient*step_size + x
        y = obj_func(x)
        convergence = abs(y - prev_y) < threshold
        prev_y = y
    #print(g_norms)
    #plt.plot(g_norms)
    #plt.title('Norm of Gradient of Quadratic Bowl')
    #plt.ylabel('Norm of gradient')
    #plt.xlabel('Iteration')
    #plt.show()
    return x

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

def stochastic_gd(data_x, data_y, tau, kappa, threshold, obj_func, make_d_func, init_theta):
    step = 0
    theta = init_theta
    prev_error = obj_func(theta)
    convergence = False
    datum_count = 0
    shuffled_deck = range(len(data_x))
    while not convergence:
        step_size = (tau + step)**(-1*kappa)
        index = shuffled_deck[datum_count]
        x = data_x[index]
        y = data_y[index]
        gradient = make_d_func(x, y)(theta)
        theta = -1*gradient*step_size + theta
        error = obj_func(theta)
        if math.isnan(error):
            return
        convergence = abs(error - prev_error) < threshold
        prev_error = error
        datum_count += 1
        if datum_count >= len(data_x):
            np.random.shuffle(shuffled_deck)
            datum_count = 0 
        #print(error)
        step += 1
    print(step)
    print("Final avg error: " + str(error))
    return theta  
if __name__ == "__main__":
    main()
