from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


'''
Returns a 100 dimension EDU embedding from a bidirectional RNN

Input: 
    x:            is in the shape (batch_size, sequence_length, embedding_size)
    seq_length:   is the length of the EDU sequence, ie. the number of words in the EDU

'''
def rnn_edu(x, seq_length)
    num_hidden = 50 # Hidden layer dimension. The RNN output dimension will be this * 2 since bidirectional_rnn will concatenate the output from forward and backward pass
    rnn_input = tf.unstack(x, seq_length, axis=1) # Unstack over axis=1 to unstack over the timesteps

    fw_cell = rnn.BasicRNNCell(num_hidden, activation=tf.nn.relu)
    bw_cell = rnn.BasicRNNCell(num_hidden, activation=tf.nn.relu)

    rnn_output, _, _ = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, rnn_input,
                                      dtype=tf.float32)
    rnn_output = rnn_output[-1] # We want the output from the last timestep only

    return rnn_output


'''
Returns a 100 dimension EDU embedding from a CNN

Input: 
    x:              is in the shape (batch_size, sequence_length, embedding_size)
    seq_length:     is the length of the EDU sequence, ie. the number of words in the EDU
    filter_size:    the number of words we want the filter to move over. For an EDU, which may be short, this should just be 2
    embedding_size: embedding dimensions, if using skip-thought embeddings this should be 4800

'''
def cnn_edu(x, seq_length, filter_size, embedding_size)
        
    cnn_input = tf.expand_dims(x, -1) # add a fourth dimension, channel, which is only 1 because we're using text not images. So cnn_input is (batch_size, sequence_length, embedding_dimension, channel)
    pooled_outputs = []
    num_filters = 100 # This will be the dimension of the output because we will max-pool over each filter, giving 100 dimensions after max-pooling
    filter_shape = [filter_size, embedding_size, 1, num_filters]

    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    conv = tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding="VALID")
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b))
    # Maxpool over the outputs
    pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
    pooled_outputs.append(pooled)
    cnn_output = tf.reshape(pooled_outputs, [-1, 100])
    
    return cnn_output
       
'''
Returns a concatenated 200 dimension EDU word embedding

Input:
    rnn_output: Should be 100 dimension embedding from rnn_edu
    cnn_output: Should be 100 dimension embedding from cnn_edu
'''
def create_edu_vector(rnn_output, cnn_output)
    return edu_vector = tf.concat([rnn_edu, cnn_edu], axis=1)
