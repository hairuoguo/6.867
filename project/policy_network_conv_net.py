import tensorflow as tf
import math
import scipy.misc
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle

def visualize_net(layer, fname):
    scipy.misc.imsave(fname, layer)

def graph_results():
    results = pickle.load(open('results.p', 'rb'))
    num_eps, avg_score, test_score = zip(*results)
    plt.plot(num_eps, avg_score, label='Training Avg Score')
    plt.plot(num_eps, test_score, label='Test Score')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('plot.png')


def weight_variable(shape, var_name):
    variable = tf.get_variable(var_name, shape, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(0.8))
    return variable

def bias_variable(shape, var_name):
    variable = tf.get_variable(var_name, shape, initializer=tf.constant_initializer(0.0), regularizer=tf.contrib.layers.l2_regularizer(0.8))
    return variable

def conv_2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def make_network(network_input, num_actions, visualize=False):
    network_input = tf.reshape(network_input, shape=[-1, 80, 80, 2])


    W_conv1 = weight_variable([4, 4, 2, 16], "W_conv1") 
    b_conv1 = bias_variable([16], "b_conv1")

    conv1 = tf.nn.relu(conv_2d(network_input, W_conv1, 2) + b_conv1) 
    #pool1 = max_pool_2x2(conv1)

    W_conv2 = weight_variable([8, 8, 16, 32], "W_conv2")
    b_conv2 = bias_variable([32], "b_conv2")

    conv2 = tf.nn.relu(conv_2d(conv1, W_conv2, 2) + b_conv2)
    #pool2 = max_pool_2x2(conv2)
    ''' 
    W_conv3 = weight_variable([3, 3, 64, 64], "W_conv3")
    b_conv3 = bias_variable([64], "b_conv3")

    conv3 = tf.nn.relu(conv_2d(conv2, W_conv3, 1) + b_conv3)
    
    shape = conv3.get_shape().as_list()
    conv3_reshaped = tf.reshape(conv3, [-1, reduce(lambda x, y: x * y, shape[1:])])
    '''

    shape = conv2.get_shape().as_list()
    conv2_reshaped = tf.reshape(conv2, [-1, reduce(lambda x, y: x * y, shape[1:])])
     
    W_fc1 = weight_variable([reduce(lambda x, y,: x*y, shape[1:]), 100], "W_fc1") 
    b_fc1 = bias_variable([100], "b_fc1")
    
    fc1 = tf.nn.relu(tf.matmul(conv2_reshaped, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([100, num_actions], "W_fc2") 
    b_fc2 = bias_variable([num_actions], "b_fc2")
    
    readout = tf.nn.softmax(tf.matmul(fc1, W_fc2) + b_fc2)

    return readout
    
def loss(readout, index):
    #difference between confidence of 1 and chosen action
    return tf.nn.sparse_softmax_cross_entropy_with_logits(readout, index)


def train(sess, env, iters, batch_size, df=0.01, visualize=False):
    network_input = tf.placeholder(tf.float32, shape=[2, 80, 80]) 
    global_step = tf.placeholder(tf.int32)
    one_hot = tf.placeholder(tf.int32, shape=[1])
    network = make_network(network_input, 2)
    one_loss = loss(network, one_hot)
    learning_rate_op = tf.maximum(0.0000001, tf.train.exponential_decay(0.01, global_step, 50000, 0.99, staircase=True))
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_op, momentum=0.95, epsilon=0.01)
    gradstep = opt.compute_gradients(one_loss)
    grads = [grad for grad, _ in gradstep]
    grads_placeholder = [(tf.placeholder(tf.float32), var) for (_, var) in gradstep]
    opt.apply_gradients(grads_placeholder)

    results = []
    pickle.dump(results, open('results.p', 'wb'))
    
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        done = True
        num_eps = 0
        reward = 0
        action_gradients = [] #list of gradients for each action taken in round
        ep_gradients = []
        ep_weights = np.array([])
        round_start_step = 0
        for step in xrange(iters):
            if reward != 0:
                weight = reward
                num_round_steps = step - round_start_step
                weights = np.array([weight*max(0, 1-(num_round_steps - n)*df) for n in xrange(num_round_steps)])
                ep_weights = np.append(ep_weights, weights)
                #ep_gradient_sum = np.sum(weights.reshape((-1, 1))*np.array(action_gradients), axis=0)
                #batch_gradients.append(ep_gradient_sum)
                round_start_step = step
                #ep_gradients.append(action_gradients)
            if done:
                obs = env.reset() 
                prev_obs = obs
                done = False
                ep_start_step = step
                if step!= 0:
                    ep_weights -= np.mean(ep_weights)
                    ep_weights /= np.std(ep_weights)
                    #ep_gradient_sum = np.sum([ep_weights[n]*ep_gradients[n] for n in xrange(len(ep_weights))], axis=0)
                    ep_gradient_sum = np.sum(ep_weights.reshape((-1, 1))*np.array(ep_gradients), axis=0)
                    batch_gradients.append(ep_gradient_sum)
                    ep_gradients = []
                    ep_weights = np.array([])
                    batch_sum += reward_sum
                    print("Ep " + str(num_eps) + " Score: " + str(reward_sum))
                if num_eps % batch_size == 0:
                    if step != 0:
                        gradients = np.sum(batch_gradients, axis=0)
                        grads_vars = [(tf.convert_to_tensor(gradient), var) for gradient, (_, var) in zip(gradients, gradstep)]
                        sess.run(opt.apply_gradients(grads_vars), feed_dict={global_step:step})
                        print("Number of rounds: " + str(num_eps))
                        print("Average score per round: " + str(batch_sum/batch_size))
                        '''run test episode'''
                        test_done = False
                        test_obs = env.reset()
                        prev_test_obs = np.zeros(obs.shape)
                        test_reward = 0.
                        test_steps = 0
                        while test_done == False:
                            test_steps += 1
                            test_frame = preprocess_frame(prev_test_obs, test_obs)
                            test_readout = sess.run(network, feed_dict={network_input: test_frame})
                            test_action = np.argmax(test_readout.flatten()) + 2
                            prev_test_obs = test_obs
                            test_obs, reward, test_done, _ = env.step(test_action)
                            test_reward += reward
                            #if test_steps >= 100:
                             #   test_done = True
                        print("Test reward: " + str(test_reward))

                        results.append((num_eps, batch_sum/batch_size, test_reward))
                    ''''''
                    batch_gradients = [] #list of summed gradients from each episode in batch
                    batch_sum = 0.
                    env.reset()
                reward_sum = 0
                if num_eps % 100 == 0:
                    saved_results = pickle.load(open('results.p', 'rb'))
                    saved_results = saved_results + results
                    pickle.dump(saved_results, open('results.p', 'wb'))
                    results = []

                    saver.save(sess, 'conv_reg', global_step=num_eps)
                num_eps += 1
            frame = preprocess_frame(prev_obs, obs)
            readout = sess.run(network, feed_dict={network_input: frame})
            #action = np.random.choice(range(env.action_space.n), p=readout.flatten())
            action = 2 if np.random.uniform() < readout.flatten()[0] else 3
            index = action - 2
            #one_hot_action = np.zeros(env.action_space.n)
            #one_hot_action[action] = 1.

            action_gradient = sess.run(grads, feed_dict={one_hot:np.array(index).reshape((1)), global_step:step, network_input:frame})
            ep_gradients.append(action_gradient)
            prev_obs = obs
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            #if step - ep_start_step >= 100: #limit the number of steps per episode (or else it might just do the same thing over and over)
             #   done = True


            #Visualize network
            if done and visualize:
                if num_eps % 1 == 0:
                    layer1 = sess.run(l1, feed_dict={network_input: frame})
                    visualize_net(layer1, "layer1_ep{}.png".format(num_eps))


def preprocess_frame(prev_frame, frame):
    prev_frame = scipy.misc.imresize(0.299*prev_frame[:, :, 0] + 0.587*prev_frame[:, :, 1] + 0.114*prev_frame[:, :, 2], (80, 80))/255.
    frame = scipy.misc.imresize(0.299*frame[:, :, 0] + 0.587*frame[:, :, 1] + 0.114*frame[:, :, 2], (80, 80))/255.
    frame_stack = np.stack([prev_frame, frame], axis=0)
    return frame_stack

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    env = gym.make('Pong-v0')
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    train(sess, env, 1000000000, 10, visualize=False)
    #graph_results()
    

if __name__ == "__main__":
    main()
