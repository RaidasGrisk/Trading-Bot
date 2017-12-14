"""
Main: script for training models
"""

import numpy as np
import pylab as pl
import tensorflow as tf
from sklearn.preprocessing import normalize
from utils import price_to_binary_target, extract_timeseries_from_oanda_data, train_test_split
from utils import remove_nan_rows, get_signal, portfolio_value, get_data_batch, get_lstm_input_output
from models import logistic_regression, vanilla_nn, lstm_nn
from get_features import get_features

# hyper-params
batch_size = 512
time_steps = 12

# load data
oanda_data = np.load('data\\EUR_GBP_H1.npy')[-10000:]
price, volume = extract_timeseries_from_oanda_data(oanda_data, ['closeMid', 'volume'])
features = get_features(oanda_data)

# prepare data
input_data_raw = np.concatenate([price, volume, features], axis=1)
output_data_raw = price_to_binary_target(oanda_data, delta=0.00025)
input_data, output_data = remove_nan_rows(input_data_raw, output_data_raw)
input_data_norm = normalize(input_data, axis=0, norm='l2')

# get dims
_, input_dim = np.shape(input_data_raw)
_, output_dim = np.shape(output_data_raw)

# split to train and test
# TODO: make a single function
train_input, test_input, train_output, test_output = train_test_split(input_data_norm, output_data, split=(0.7, 0.3))

# forward-propagation
# x, y, logits, y_ = logistic_regression(input_dim, output_dim, drop_keep_prob=0.9)
# x, y, logits, y_ = vanilla_nn(input_dim, output_dim, [64, 128, 64, 16], drop_layer=3, drop_keep_prob=0.9)
x, y, logits, y_ = lstm_nn(input_dim, output_dim, time_steps=time_steps, n_hidden=[64, 32, 8], drop_keep_prob=0.5)

# tf cost and optimizer
# TODO: maximize return or sharpe or something, but not cross-entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

# init session
step, cost_hist, value_hist = 0, [], []
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
while True:

    # train model
    train_x, train_y, indexes = get_data_batch(train_input, train_output, batch_size)
    train_x, train_y = get_lstm_input_output(train_x, train_y, time_steps=time_steps)
    _, train_cost = sess.run([train_step, cost], feed_dict={x: train_x, y: train_y})

    # keep track of stuff
    step += 1
    if step % 10 == 0 or step == 1:

        # # get y_ predictions
        # y_train = sess.run([y_], feed_dict={x: train_input, y: train_output})
        # test_cost, y_test = sess.run([cost, y_], feed_dict={x: test_input, y: test_output})
        #
        # # get portfolio value
        # signal_test, signal_train = get_signal(y_test), get_signal(y_train[0])
        # value_test = portfolio_value(test_input[:, 0], signal_test, trans_costs=0)
        # value_train = portfolio_value(train_input[:, 0], signal_train, trans_costs=0)
        #
        # cost_hist.append([train_cost / batch_size, test_cost / batch_size])
        # value_hist.append([value_train[-1], value_test[-1]])
        #
        # pl.figure(1)
        # pl.subplot(211)
        # pl.plot(cost_hist)
        # pl.subplot(212)
        # pl.plot(value_hist)
        # pl.pause(1e-5)
        #
        # print('Train cost: {:.4f}, Test cost: {:.4f},  Test value: {:.2f}'
        #       .format(train_cost, test_cost, value_hist[-1][1]))



        # if lstm
        if step == 1:
            train_input_lstm, train_output_lst = get_lstm_input_output(train_input, train_output, time_steps=time_steps)
            test_input_lstm, test_output_lst = get_lstm_input_output(test_input, test_output, time_steps=time_steps)

        # get y_ predictions
        y_train = sess.run([y_], feed_dict={x: train_input_lstm, y: train_output_lst})
        test_cost, y_test = sess.run([cost, y_], feed_dict={x: test_input_lstm, y: test_output_lst})

        # get portfolio value
        signal_test, signal_train = get_signal(y_test), get_signal(y_train[0])
        value_test = portfolio_value(test_input[time_steps:, 0], signal_test, trans_costs=0)
        value_train = portfolio_value(train_input[time_steps:, 0], signal_train, trans_costs=0)

        cost_hist.append([train_cost / batch_size, test_cost / batch_size])
        value_hist.append([value_train[-1], value_test[-1]])

        pl.figure(1)
        pl.subplot(211)
        pl.plot(cost_hist)
        pl.subplot(212)
        pl.plot(value_hist)
        pl.pause(1e-5)

        print('Train cost: {:.4f}, Test cost: {:.4f},  Test value: {:.2f}'
              .format(train_cost, test_cost, value_hist[-1][1]))