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
    variable = tf.get_variable(var_name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return variable

def bias_variable(shape, var_name):
    variable = tf.get_variable(var_name, shape, initializer=tf.constant_initializer(0.0))
    return variable

def make_network(network_input, num_actions):
    W1 = weight_variable([80*80, 200], "W1") 
    b1 = bias_variable([200], "b1")
    l1 = tf.nn.relu(tf.matmul(network_input, W1) + b1, "l1")

    W2 = weight_variable([200, num_actions], "W2")
    b2 = bias_variable([num_actions], "b2")
    readout = tf.nn.softmax(tf.matmul(l1, W2) + b2)

    return readout

def loss(readout, index):
    #difference between confidence of 1 and chosen action
    return tf.nn.sparse_softmax_cross_entropy_with_logits(readout, index)


def train(sess, env, iters, batch_size, df=0.01, visualize=False):
    network_input = tf.placeholder(tf.float32, shape=[1, 80*80]) 
    global_step = tf.placeholder(tf.int32)
    one_hot = tf.placeholder(tf.int32, shape=[1])
    network = make_network(network_input, 2)
    one_loss = loss(network, one_hot)
    learning_rate_op = tf.maximum(0.00001, tf.train.exponential_decay(0.001, global_step, 50000, 0.99, staircase=True))
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate_op, momentum=0.95, epsilon=0.01)
    gradstep = opt.compute_gradients(one_loss)
    grads = [grad for grad, _ in gradstep]
    grads_placeholder = [(tf.placeholder(tf.float32), var) for (_, var) in gradstep]
    opt.apply_gradients(grads_placeholder)

    results = []
    #pickle.dump(results, open('results.p', 'wb'))
    
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "my_model-3000")
        done = True
        num_eps = 3000
        reward = 0
        action_gradients = [] #list of gradients for each action taken in round
        for step in xrange(iters):
            if reward != 0:
                weight = reward
                weights = np.array([weight*max(0, 1-(len(action_gradients) - n)*df) for n in xrange(len(action_gradients))])
                ep_gradient_sum = np.sum(weights.reshape((-1, 1))*np.array(action_gradients), axis=0)
                batch_gradients.append(ep_gradient_sum)
                action_gradients = [] #list of gradients for each action taken in round
            if done:
                obs = env.reset()
                prev_obs = np.zeros(obs.shape)
                done = False
                ep_start_step = step
                if step!= 0: 
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

                        #results.append((num_eps, batch_sum/batch_size, test_reward))
                    ''''''
                    batch_gradients = [] #list of summed gradients from each episode in batch
                    batch_sum = 0.
                    env.reset()
                reward_sum = 0
                if num_eps % 100 == 0:
                    # load saved results, and append new results
                    '''
                    saved_results = pickle.load(open('results.p', 'rb'))
                    saved_results = saved_results + results
                    pickle.dump(saved_results, open('results.p', 'wb'))
                    results = []
                    '''
                    saver.save(sess, 'my_model', global_step=num_eps)
                    
                num_eps += 1
            frame = preprocess_frame(prev_obs, obs)
            readout = sess.run(network, feed_dict={network_input: frame})

            #action = np.random.choice(range(env.action_space.n), p=readout.flatten())
            action = 2 if np.random.uniform() < readout.flatten()[0] else 3
            index = action - 2
            #one_hot_action = np.zeros(env.action_space.n)
            #one_hot_action[action] = 1.

            action_gradient = sess.run(grads, feed_dict={one_hot:np.array(index).reshape((1)), global_step:step, network_input:frame})
            action_gradients.append(action_gradient)
            prev_obs = obs
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            #if step - ep_start_step >= 100: #limit the number of steps per episode (or else it might just do the same thing over and over)
             #   done = True
'''
            break;
            # Visualize network
            if done and visualize:
                if num_eps % 1 == 0: # do this every n eps?
                    weights1, layer1, weights2 = sess.run([W1, l1, W2], feed_dict={network_input: frame})
                    visualize_net(layer1, "layer1_ep{}.png".format(num_eps))
                    visualize_net(weights1, "weights1_ep{}.png".format(num_eps))
                    visualize_net(weights2, "weights2_ep{}.png".format(num_eps))

                    # take column slice of [80*80, 200] weights1 and reshape into [80, 80], so it looks like frame
                    column_number = 0 # can be any of 200 slices 
                    frame_weights1 = weights1[:, column_number].reshape((80, 80))
                    visualize_net(frame_weights1, "frame1_ep{}.png".format(num_eps))
                    # break;
'''

def preprocess_frame(prev_frame, frame):
    difference = scipy.misc.imresize((frame-prev_frame)/255., (80, 80))
    difference = 0.299*difference[:, :, 0] + 0.587*difference[:, :, 1] + 0.114*difference[:, :, 2]
    return difference.reshape((1, -1))

def main():
    env = gym.make('Pong-v0')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #network_input = tf.placeholder(tf.float32, shape=[80*80, 1]) 
    train(sess, env, 1000000000, 10)
    

if __name__ == "__main__":
    main()
