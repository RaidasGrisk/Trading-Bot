"""
Training logistic regression v2:
using regression to allocate funds.

Things to work out:
1. price_data_raw percentage or pips?
2. objective function normalization (yearly percentage return..? etc)


"""

import numpy as np
import pylab as pl
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from helpers.utils import extract_timeseries_from_oanda_data, train_test_validation_split
from helpers.utils import remove_nan_rows, get_data_batch
from models import logistic_regression
from helpers.get_features import get_features

# hyper-params
batch_size = 1024
plotting = False

# load data
# TODO: check shit np.concatenate([price, input_data_raw[:, 0:1], price_data_raw[:, 0:1], output_data_raw], axis=1)
# TODO np.set_printoptions(linewidth=75*3+5, edgeitems=6)
oanda_data = np.load('data\\AUD_USD_H1.npy')
price_data_raw = extract_timeseries_from_oanda_data(oanda_data, ['closeMid'])
input_data_raw, input_data_dummy = get_features(oanda_data)
# price_data_raw = np.concatenate([[[0]], price_data_raw[1:] - price_data_raw[:-1]], axis=0)
# TODO: new price data
price_data_raw = np.concatenate([[[0]], (price_data_raw[1:] - price_data_raw[:-1]) / (price_data_raw[1:] + 1e-10)], axis=0)

# prepare data
input_data, price_data, input_data_dummy = remove_nan_rows([input_data_raw, price_data_raw, input_data_dummy])
input_data_scaled = np.concatenate([minmax_scale(input_data, axis=0), input_data_dummy], axis=1)

# split to train,test and cross validation
input_train, input_test, input_cv, price_train, price_test, price_cv = \
    train_test_validation_split([input_data_scaled, price_data], split=(0.5, 0.3, 0.2))

# get dims
_, input_dim = np.shape(input_data_scaled)

# forward-propagation
x, _, logits, y_ = logistic_regression(input_dim, 3, drop_keep_prob=0.8)

# tf cost and optimizer
# TODO: maximize return or sharpe or something, but not cross-entropy
price_h = tf.placeholder(tf.float32, [None, 1])
signals = tf.constant([[1., -1., 0.]])
objective = (tf.reduce_mean(y_[:-1] * signals * price_h[1:]) + tf.constant(1.))  # profit function
train_step = tf.train.AdamOptimizer(0.001).minimize(-objective)

# init session
step, cost_hist_train, cost_hist_test, value_hist_train, value_hist_test, value_hist_cv, value_hist_cv_ma = \
    0, [], [], [], [], [], []
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# main loop
while True:

    # train model
    x_train, price_batch = get_data_batch([input_train, price_train], batch_size)
    _, cost_train, sig = sess.run([train_step, objective, y_], feed_dict={x: x_train, price_h: price_batch})

    # keep track of stuff
    step += 1
    if step % 10 == 0 or step == 1:

        # get y_ predictions
        y_train_pred = sess.run(y_, feed_dict={x: input_train})
        y_test_pred, cost_test = sess.run([y_, objective], feed_dict={x: input_test, price_h: price_test})
        y_cv_pred = sess.run(y_, feed_dict={x: input_cv})

        # get portfolio value
        value_test = np.cumsum(np.sum(y_test_pred[:-1] * [1., -1., 0.] * price_test[1:], axis=1))
        value_train = np.cumsum(np.sum(y_train_pred[:-1] * [1., -1., 0.] * price_train[1:], axis=1))
        value_cv = np.cumsum(np.sum(y_cv_pred[:-1] * [1., -1., 0.] * price_cv[1:], axis=1))

        # save history
        cost_hist_train.append(cost_train)
        cost_hist_test.append(cost_test)

        print('Step {}: train {:.4f}, test {:.4f}'.format(step, cost_train, cost_test))

        if plotting:

            pl.figure(1)
            pl.title('Cost')
            pl.plot(cost_hist_train, color='darkorange')
            pl.plot(cost_hist_test, color='dodgerblue')

            pl.pause(1e-10)

            if value_test[-1] > 0.01 and value_train[-1] > 0.01 and value_cv[-1] > 0.01:
                print(value_train[-1], value_test[-1], value_cv[-1])

                pl.figure(2)
                pl.plot(value_test)
                pl.plot(value_train)
                pl.plot(value_cv)


