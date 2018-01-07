"""
A set of models to train
"""

import tensorflow as tf
import numpy as np


def logistic_regression(input_dim, output_dim):
    """Simple logistic regression
    Returns x and y placeholders, logits and y_ (y hat)"""
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, input_dim])
    y = tf.placeholder(tf.float32, [None, output_dim])
    learning_r = tf.placeholder(tf.float32, 1)[0]
    drop_out = tf.placeholder(tf.float32, 1)[0]

    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.initializers.truncated_normal(mean=0.1, stddev=0.025)
    w = tf.get_variable('weights1', shape=[input_dim, output_dim], initializer=w_init)
    b = tf.get_variable('bias1', shape=[output_dim], initializer=b_init)

    logits = tf.matmul(tf.nn.dropout(x, keep_prob=drop_out), w) + b
    y_ = tf.nn.softmax(logits)

    [print(var) for var in tf.trainable_variables()]
    return x, y, logits, y_, learning_r, drop_out


def lstm_nn(input_dim, output_dim, time_steps, n_hidden):
    """LSTM net returns x and y placeholders, logits and y_ (y hat)"""

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, time_steps, input_dim])
    y = tf.placeholder(tf.float32, [None, output_dim])
    learning_r = tf.placeholder(tf.float32, 1)[0]
    drop_out = tf.placeholder(tf.float32, 1)[0]

    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.initializers.truncated_normal(mean=0.1, stddev=0.025)
    w = tf.get_variable('last_weights', shape=[n_hidden[-1], output_dim], initializer=w_init)
    # b = tf.get_variable('bias1', shape=[output_dim], initializer=b_init)

    x_split = tf.unstack(x, time_steps, 1)

    # stack lstm cells, a cell per hidden layer
    stacked_lstm_cells = []  # a list of lstm cells to be inputted into MultiRNNCell
    for layer_size in n_hidden:
        stacked_lstm_cells.append(tf.contrib.rnn.BasicLSTMCell(layer_size, activation=tf.nn.relu))

    # create the net and add dropout
    lstm_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm_cells)
    lstm_cell_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=drop_out)

    # forward propagate
    outputs, state = tf.contrib.rnn.static_rnn(lstm_cell_with_dropout, x_split, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], w)  # + b  # logits are used for cross entropy
    y_ = tf.nn.softmax(logits)

    [print(var) for var in tf.trainable_variables()]
    print([print(i) for i in outputs])
    print(y_)
    return x, y, logits, y_, learning_r, drop_out


def cnn(input_dim, output_dim, time_steps, filter):
    """CNN returns x and y placeholders, logits and y_ (y hat)"""

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, input_dim, time_steps, 1])
    y = tf.placeholder(tf.float32, [None, output_dim])
    learning_r = tf.placeholder(tf.float32, 1)[0]
    drop_out = tf.placeholder(tf.float32, 1)[0]

    conv1 = tf.layers.conv2d(inputs=x,
                             filters=filter[0],
                             kernel_size=(input_dim, 1),
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                             strides=1,
                             padding='valid',
                             activation=tf.nn.relu)
    conv1_dropout = tf.layers.dropout(inputs=conv1, rate=drop_out)
    conv2 = tf.layers.conv2d(inputs=conv1_dropout,
                             filters=filter[1],
                             kernel_size=(1, time_steps),
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                             strides=1,
                             padding='valid',
                             activation=tf.nn.relu)
    logits_dense = tf.layers.dense(inputs=conv2,
                                   units=output_dim,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                   activation=None,
                                   use_bias=False)

    logits = tf.reshape(logits_dense, (-1, output_dim))
    y_ = tf.nn.softmax(tf.reshape(logits_dense, (-1, output_dim)))

    [print(var) for var in tf.trainable_variables()]
    print(y_)
    return x, y, logits, y_, learning_r, drop_out


def vanilla_nn(input_dim, output_dim, architecture, drop_layer=0, drop_keep_prob=0.9):
    """Vanilla neural net
    Returns x and y placeholders, logits and y_ (y hat)"""
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, input_dim])
    y = tf.placeholder(tf.float32, [None, output_dim])

    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.initializers.truncated_normal(mean=0.1, stddev=0.025)
    layer_sizes = [input_dim] + architecture + [output_dim]
    weights, biases = {}, {}
    layer_values = {0: x}
    for layer, size_current, size_next in zip(range(len(layer_sizes)), layer_sizes, layer_sizes[1:]):

        # create weights
        last_layer = layer == len(layer_sizes) - 2  # dummy variable for last layer
        weights[layer] = tf.get_variable('weights{}'.format(layer), shape=[size_current, size_next], initializer=w_init)
        biases[layer] = tf.get_variable('biases{}'.format(layer), shape=[size_next], initializer=b_init)

        # forward-propagate
        if not last_layer:
            layer_values[layer+1] = tf.nn.relu(tf.matmul(layer_values[layer], weights[layer]) + biases[layer])
        else:
            layer_values[layer+1] = tf.matmul(layer_values[layer], weights[layer]) + biases[layer]
            y_ = tf.nn.softmax(layer_values[layer+1])
        if drop_layer == layer:
            layer_values[layer+1] = tf.nn.dropout(layer_values[layer+1], keep_prob=drop_keep_prob)

    [print(var) for var in tf.trainable_variables()]
    print([print(value) for _, value in layer_values.items()])
    print(y_)
    return x, y, layer_values[len(layer_values)-1], y_

