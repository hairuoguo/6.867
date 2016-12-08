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

def make_network(network_input, num_actions, visualize=False):
    network_input = tf.reshape(network_input, shape=[-1, 80, 80, 1])
    W1 = weight_variable([80, 80, 1, 200], "W1") 
    b1 = bias_variable([200], "b1")
    conv1 = tf.nn.conv2d(network_input, W1, strides=[1,1,1,1], padding="VALID", name="conv1")
    l1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1") 

    W2 = weight_variable([200, num_actions], "W2") 
    b2 = bias_variable([num_actions], "b2")

    # l1 reshape is (1, 200)
    l1 = tf.reshape(l1, [-1, W2.get_shape().as_list()[0]]) 
    readout = tf.nn.softmax(tf.matmul(l1, W2) + b2)

    return l1, readout

def loss(readout, index):
    #difference between confidence of 1 and chosen action
    return tf.nn.sparse_softmax_cross_entropy_with_logits(readout, index)


def train(sess, env, iters, batch_size, df=0.01, visualize=False):
    network_input = tf.placeholder(tf.float32, shape=[80, 80]) 
    global_step = tf.placeholder(tf.int32)
    one_hot = tf.placeholder(tf.int32, shape=[1])
    l1, network = make_network(network_input, 2)
    one_loss = loss(network, one_hot)
    learning_rate_op = tf.maximum(0.00001, tf.train.exponential_decay(0.0001, global_step, 50000, 0.99, staircase=True))
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
                            test_action = 2 if np.random.uniform() < test_readout.flatten()[0] else 3
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


            #Visualize network
            if done and visualize:
                if num_eps % 1 == 0:
                    layer1 = sess.run(l1, feed_dict={network_input: frame})
                    visualize_net(layer1, "layer1_ep{}.png".format(num_eps))


def preprocess_frame(prev_frame, frame):
    difference = scipy.misc.imresize((frame-prev_frame)/255., (80, 80))
    difference = 0.299*difference[:, :, 0] + 0.587*difference[:, :, 1] + 0.114*difference[:, :, 2]
    return difference

def main():
    env = gym.make('Pong-v0')
    sess = tf.Session()
    train(sess, env, 10000000, 10, visualize=False)
    #graph_results()
    

if __name__ == "__main__":
    main()
