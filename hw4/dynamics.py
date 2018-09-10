import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.action_size = len(env.action_space.sample())
        self.observation_size = len(env.observation_space.high)
        self.input_placeholder = tf.placeholder(tf.float32, [None, self.action_size + self.observation_size])
        self.labels_placeholder = tf.placeholder(tf.float32, [None, self.observation_size])
        self.environment = env
        self.output = build_mlp(self.input_placeholder, self.observation_size, "dynamics_model", n_layers, size, activation, output_activation)
        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.loss = tf.square(self.output - self.labels_placeholder)
        self.step = self.optimizer.minimize(self.loss)


    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        states = (data['states'] - self.normalization['states_mean']) / (self.normalization['states_stdev'] + 1e-7)
        actions = (data['actions'] - self.normalization['actions_mean']) / (self.normalization['actions_stdev'] + 1e-7)

        input_values = np.concatenate((states, actions), axis = 1)

        deltas = (data['deltas'] - self.normalization['deltas_mean']) / (self.normalization['deltas_stdev'] + 1e-7)

        self.sess.run(self.step, feed_dict={self.input_placeholder : input_values, self.labels_placeholder : deltas})





    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        input_states = (states - self.normalization['states_mean']) / (self.normalization['states_stdev'] + 1e-7)
        input_actions = (actions - self.normalization['actions_mean']) / (self.normalization['actions_stdev'] + 1e-7)

        input_values = np.concatenate((input_states, input_actions), axis = 1)

        deltas_normalized = self.sess.run(self.output, feed_dict = {self.input_placeholder : input_values})

        deltas = deltas_normalized * self.normalization['deltas_stdev'] + (self.normalization['deltas_mean'] + 1e-7)

        return input_states + deltas