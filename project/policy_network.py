import tensorflow as tf
import math
import scipy as sp
import numpy as np
import gym


def weight_variable(shape, var_name):
    variable = tf.get_variable(var_name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return variable

def bias_variable(shape, var_name):
    variable = tf.get_variable(var_name, shape, initializer=tf.constant_initializer(0.0))
    return variable

def make_network(network_input, num_actions):
    W1 = weight_variable([1, 200], "W1") 
    b1 = bias_variable([200], "b1")
    l1 = tf.nn.relu(tf.matmul(network_input, W1) + b1)

    W2 = weight_variable([200, num_actions], "W2")
    b2 = bias_variable([num_actions], "b2")
    
    readout = tf.nn.softmax(tf.matmul(l1, W2) + b2)

    return readout

def loss(readout, index):
    #difference between confidence of 1 and chosen action
    return tf.nn.sparse_softmax_cross_entropy_with_logits(readout, index)


def train(sess, env, iters, batch_size, df=0.01):
    network_input = tf.placeholder(tf.float32, shape=[1, 1]) 
    global_step = tf.placeholder(tf.int32)
    one_hot = tf.placeholder(tf.int32, shape=[1])
    network = make_network(network_input, env.action_space.n)
    one_loss = loss(network, one_hot)
    learning_rate_op = tf.maximum(0.00001, tf.train.exponential_decay(0.001, global_step, 50000, 0.95, staircase=True))
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_op, momentum=0.95, epsilon=0.01)
    gradstep = opt.compute_gradients(one_loss)
    grads = [grad for grad, _ in gradstep]
    grads_placeholder = [(tf.placeholder(tf.float32), var) for (_, var) in gradstep]
    opt.apply_gradients(grads_placeholder)
    
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        done = True
        num_rounds = 0
        for step in xrange(iters):
            if done:
                obs = env.reset()
                prev_obs = np.zeros(obs.shape)
                done = False
                ep_start_step = step
                if step != 0:
                    if reward_sum == 0.:
                    #positive or negative gradient based on win or lose
                        weight = -1.
                    else:
                        weight = 1.
                    weights = np.array([weight*max(0, 1-(len(action_gradients) - n)*df) for n in xrange(len(action_gradients))])
                    ep_gradient_sum = np.sum(weights.reshape((-1, 1))*np.array(action_gradients), axis=0)
                    batch_gradients.append(ep_gradient_sum)
                    batch_sum += reward_sum
                if num_rounds % batch_size == 0:
                    if step != 0:
                        gradients = np.sum(batch_gradients, axis=0)
                        grads_vars = [(tf.convert_to_tensor(gradient), var) for gradient, (_, var) in zip(gradients, gradstep)]
                        sess.run(opt.apply_gradients(grads_vars), feed_dict={global_step:step})
                        print("Number of rounds: " + str(num_rounds))
                        print("Average score per round: " + str(batch_sum/batch_size))
                        '''run test episode'''
                        test_done = False
                        test_obs = env.reset()
                        prev_test_obs = np.zeros(obs.shape)
                        test_reward = 0.
                        test_steps = 0
                        test_states = []
                        test_actions = []
                        while test_done == False:
                            test_steps += 1
                            test_frame = preprocess_frame(prev_test_obs, test_obs)
                            test_readout = sess.run(network, feed_dict={network_input: test_frame})
                            print(test_readout.flatten())
                            test_action = np.argmax(test_readout.flatten())
                            test_actions.append(test_action)
                            prev_test_obs = test_obs
                            test_obs, reward, test_done, _ = env.step(test_action)
                            test_states.append(test_obs)
                            test_reward += reward
                            if test_steps >= 10:
                                test_done = True
                        print(test_states)
                        print(test_actions)
                        print("Test reward: " + str(test_reward))
                    ''''''
                    batch_gradients = [] #list of summed gradients from each episode in batch
                    batch_sum = 0.
                    env.reset()
                reward_sum = 0
                action_gradients = [] #list of gradients for each action taken in batch
                num_rounds += 1
            frame = preprocess_frame(prev_obs, obs)
            readout = sess.run(network, feed_dict={network_input: frame})
            action = np.random.choice(range(env.action_space.n), p=readout.flatten())
            #one_hot_action = np.zeros(env.action_space.n)
            #one_hot_action[action] = 1.

            action_gradient = sess.run(grads, feed_dict={one_hot:action.reshape((1)), global_step:step, network_input:frame})
            action_gradients.append(action_gradient)
            prev_obs = obs
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if step - ep_start_step >= 10: #limit the number of steps per episode (or else it might just do the same thing over and over)
                done = True


def preprocess_frame(prev_frame, frame):
    '''
    difference = sp.misc.imresize((frame-prev_frame)/255., (80, 80))
    difference = 0.299*difference[:, :, 0] + 0.587*difference[:, :, 1] + 0.114*difference[:, :, 2]
    return difference.ravel()
    '''
    return np.array([frame]).reshape((1, 1))

def main():
    env = gym.make('FrozenLake-v0')
    sess = tf.Session()
    #network_input = tf.placeholder(tf.float32, shape=[80*80, 1]) 
    train(sess, env, 10000000, 100)
    

if __name__ == "__main__":
    main()
