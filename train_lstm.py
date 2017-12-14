"""
Training lstm model
"""

import numpy as np
import pylab as pl
import tensorflow as tf
from sklearn.preprocessing import normalize
from helpers.utils import price_to_binary_target, extract_timeseries_from_oanda_data, train_test_validation_split
from helpers.utils import remove_nan_rows, get_signal, portfolio_value, get_data_batch, get_lstm_input_output
from models import lstm_nn
from helpers.get_features import get_features

# hyper-params
batch_size = 1024
time_steps = 6 * 12
plotting = False
value_cv_moving_average = 100
split = (0.9, 0.05, 0.05)

# load data
oanda_data = np.load('data\\AUD_USD_M10.npy')
input_data_raw = get_features(oanda_data)
output_data_raw = price_to_binary_target(oanda_data, delta=0.00025)
price_data_raw = extract_timeseries_from_oanda_data(oanda_data, ['closeMid'])

# prepare data
input_data, output_data, price_data = remove_nan_rows([input_data_raw, output_data_raw, price_data_raw])
input_data = normalize(input_data, axis=1, norm='l2')
input_data, output_data = get_lstm_input_output(input_data, output_data, time_steps=time_steps)
price_data = price_data[-len(input_data):]

# split to train,test and cross validation
input_train, input_test, input_cv, output_train, output_test, output_cv = \
    train_test_validation_split(input_data, output_data, split=split)
price_train, price_test, price_cv, _, _, _ = train_test_validation_split(price_data, price_data, split=split)

# get dims
_, _, input_dim = np.shape(input_data)
_, output_dim = np.shape(output_data)

# forward-propagation
x, y, logits, y_ = lstm_nn(input_dim, output_dim, time_steps=time_steps, n_hidden=[32], drop_keep_prob=0.7)

# tf cost and optimizer
# TODO: maximize return or sharpe or something, but not cross-entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

# init session
step, cost_hist_train, cost_hist_test, value_hist_train, value_hist_test, value_hist_cv, value_hist_cv_ma = \
    0, [], [], [], [], [], []
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
while True:

    # train model
    x_train, y_train = get_data_batch(input_train, output_train, batch_size)
    _, cost_train = sess.run([train_step, cost], feed_dict={x: x_train, y: y_train})

    # keep track of stuff
    step += 1
    if step % 10000 == 0 or step == 1:

        # get y_ predictions
        y_train = sess.run([y_], feed_dict={x: input_train, y: output_train})
        cost_test, y_test = sess.run([cost, y_], feed_dict={x: input_test, y: output_test})
        y_cv = sess.run([y_], feed_dict={x: input_cv, y: output_cv})

        # get portfolio value
        signal_test, signal_train, signal_cv = get_signal(y_test), get_signal(y_train[0]), get_signal(y_cv[0])
        value_test = portfolio_value(price_test, signal_test, trans_costs=0)
        value_train = portfolio_value(price_train, signal_train, trans_costs=0)
        value_cv = portfolio_value(price_cv, signal_cv, trans_costs=0)

        # save history
        cost_hist_train.append(cost_train / batch_size)
        cost_hist_test.append(cost_test / batch_size)
        value_hist_train.append(value_train[-1])
        value_hist_test.append(value_test[-1])
        value_hist_cv.append(value_cv[-1])
        value_hist_cv_ma.append(np.mean(value_hist_cv[-value_cv_moving_average:]))

        print('Step {}: train {:.4f}, test {:.4f}'.format(step, cost_train, cost_test))

        if plotting:

            pl.figure(1)

            pl.subplot(211)
            pl.title('Cost')
            pl.plot(cost_hist_train, color='darkorange')
            pl.plot(cost_hist_test, color='dodgerblue')

            pl.subplot(212)
            pl.title('Portfolio value')
            pl.plot(value_hist_train, color='darkorange', linewidth=0.2)
            pl.plot(value_hist_test, color='dodgerblue', linewidth=0.2)
            pl.plot(value_hist_cv, color='magenta', linewidth=2)
            pl.plot(value_hist_cv_ma, color='black', linewidth=0.5)

            pl.pause(1e-10)

pl.figure(3)
pl.plot(value_train)
pl.plot(value_test)
pl.plot(value_cv)
