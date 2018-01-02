"""
Training lstm v2:
using model to allocate funds, i.e. maximizing return without correct labels.

Other options to set:
np.set_printoptions(linewidth=75*3+5, edgeitems=6)
pl.rcParams.update({'font.size': 6})
"""

import numpy as np
import pylab as pl
import tensorflow as tf
from helpers.utils import extract_timeseries_from_oanda_data, train_test_validation_split
from helpers.utils import remove_nan_rows, get_data_batch, get_lstm_input_output
from models import lstm_nn
from helpers.get_features import get_features, min_max_scaling

# hyper-params
batch_size = 256
time_steps = 12
plotting = False
saving = False
value_cv_moving_average = 50
split = (0.6, 0.2, 0.1)

# load data
oanda_data = np.load('data\\EUR_USD_M10.npy')  # [-50000:]
price_data_raw = extract_timeseries_from_oanda_data(oanda_data, ['closeMid'])
input_data_raw, input_data_dummy = get_features(oanda_data)
price_data_raw = np.concatenate([[[0]],
                                 (price_data_raw[1:] - price_data_raw[:-1]) / (price_data_raw[1:] + 1e-10)], axis=0)

# prepare data
input_data, price_data, input_data_dummy = remove_nan_rows([input_data_raw, price_data_raw, input_data_dummy])
input_data_scaled_no_dummies = (input_data - min_max_scaling[1, :]) / (min_max_scaling[0, :] - min_max_scaling[1, :])
input_data_scaled = np.concatenate([input_data_scaled_no_dummies, input_data_dummy], axis=1)
input_data_lstm, _ = get_lstm_input_output(input_data_scaled, np.zeros_like(input_data), time_steps=time_steps)
price_data = price_data[-len(input_data_lstm):]

# split to train,test and cross validation
input_train, input_test, input_cv, price_train, price_test, price_cv = \
    train_test_validation_split([input_data_lstm, price_data], split=split)

# get dims
_, _, input_dim = np.shape(input_train)

# forward-propagation
x, y, logits, y_ = lstm_nn(input_dim, 3, time_steps=time_steps, n_hidden=[3], drop_keep_prob=0.6)

# tf cost and optimizer
price_h = tf.placeholder(tf.float32, [None, 1])
signals = tf.constant([[1., -1., 0.]])
objective = (tf.reduce_mean(y_[:-1] * signals * price_h[1:] * 100))  # profit function
train_step = tf.train.AdamOptimizer(0.001).minimize(-objective)


# init session
step, step_hist, objective_hist_train, objective_hist_test, value_hist_train, value_hist_test, \
    value_hist_cv, value_hist_cv_ma, saving_score = 0, [], [], [], [], [], [], [], 0.05
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
while True:

    # train model
    x_train, price_batch = get_data_batch([input_train, price_train], batch_size)
    _, objective_train, sig = sess.run([train_step, objective, y_], feed_dict={x: x_train, price_h: price_batch})

    # keep track of stuff
    step += 1
    if step % 100 == 0 or step == 1:

        # get y_ predictions
        y_train_pred = sess.run(y_, feed_dict={x: input_train})
        y_test_pred, objective_test = sess.run([y_, objective], feed_dict={x: input_test, price_h: price_test})
        y_cv_pred = sess.run(y_, feed_dict={x: input_cv})

        # get portfolio value
        value_train = 1 + np.cumsum(np.sum(y_train_pred[:-1] * [1., -1., 0.] * price_train[1:], axis=1))
        value_test = 1 + np.cumsum(np.sum(y_test_pred[:-1] * [1., -1., 0.] * price_test[1:], axis=1))
        value_cv = 1 + np.cumsum(np.sum(y_cv_pred[:-1] * [1., -1., 0.] * price_cv[1:], axis=1))

        # save history
        step_hist.append(step)
        objective_hist_train.append(objective_train)
        objective_hist_test.append(objective_test)
        value_hist_train.append(value_train[-1])
        value_hist_test.append(value_test[-1])
        value_hist_cv.append(value_cv[-1])
        value_hist_cv_ma.append(np.mean(value_hist_cv[-value_cv_moving_average:]))

        print('Step {}: train {:.4f}, test {:.4f}'.format(step, objective_train, objective_test))

        if plotting:

            pl.figure(1, figsize=(3, 7), dpi=80, facecolor='w', edgecolor='k')

            pl.subplot(211)
            pl.title('Objective function')
            pl.plot(step_hist, objective_hist_train, color='darkorange', linewidth=0.3)
            pl.plot(step_hist, objective_hist_test, color='dodgerblue', linewidth=0.3)

            pl.subplot(212)
            pl.title('Portfolio value')
            pl.plot(step_hist, value_hist_train, color='darkorange', linewidth=0.3)
            pl.plot(step_hist, value_hist_test, color='dodgerblue', linewidth=0.3)
            pl.plot(step_hist, value_hist_cv, color='magenta', linewidth=1)
            pl.plot(step_hist, value_hist_cv_ma, color='black', linewidth=0.5)
            pl.pause(1e-10)

        # save if some complicated rules
        if saving:
            current_score = 0 if value_test[-1] < 0.01 or value_cv[-1] < 0.01 \
                else np.average([value_test[-1], value_cv[-1]])
            saving_score = current_score if saving_score < current_score else saving_score
            if saving_score == current_score and saving_score > 0.05:
                saver.save(sess, 'saved_models/lstm-v2-avg_score{:.3f}'.format(current_score), global_step=step)
                print('Model saved. Average score: {:.2f}'.format(current_score))

                pl.figure(2)
                pl.plot(value_test, linewidth=0.2)
                pl.plot(value_cv, linewidth=2)
                pl.pause(1e-10)

