"""
A set of models to train
"""

import tensorflow as tf


def logistic_regression(input_dim, output_dim, drop_keep_prob):
    """Simple logistic regression
    Returns x and y placeholders, logits and y_ (y hat)"""
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, input_dim])
    y = tf.placeholder(tf.float32, [None, output_dim])

    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.initializers.truncated_normal(mean=0.1, stddev=0.025)
    weights = {0: tf.get_variable('weights1', shape=[input_dim, output_dim], initializer=w_init)}
    biases = {0: tf.get_variable('bias1', shape=[output_dim], initializer=b_init)}

    logits = tf.nn.dropout(tf.matmul(x, weights[0]) + biases[0], keep_prob=drop_keep_prob)
    y_ = tf.nn.softmax(logits)

    [print(var) for var in tf.trainable_variables()]
    return x, y, logits, y_


def vanilla_nn(input_dim, output_dim, architecture, drop_layer=2, drop_keep_prob=0.9):
    """Vanilla neural net
    Returns x and y placeholders, logits and y_ (y hat)"""

    architecture = [64, 32, 16, 8]

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
        if not last_layer:
            biases[layer] = tf.get_variable('biases{}'.format(layer), shape=[size_next], initializer=b_init)

        # forward-propagate
        if not last_layer:
            layer_values[layer+1] = tf.nn.relu(tf.matmul(layer_values[layer], weights[layer]) + biases[layer])
        else:
            layer_values[layer+1] = tf.matmul(layer_values[layer], weights[layer])
            y_ = tf.nn.softmax(layer_values[layer+1])
        if drop_layer == layer:
            layer_values[layer+1] = tf.nn.dropout(layer_values[layer+1], keep_prob=drop_keep_prob)

    [print(var) for var in tf.trainable_variables()]
    print([print(value) for _, value in layer_values.items()])
    print(y_)
    return x, y, layer_values[len(layer_values)-1], y_
