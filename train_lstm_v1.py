"""
Training lstm model v1
"""

import numpy as np
import pylab as pl
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from helpers.utils import price_to_binary_target, extract_timeseries_from_oanda_data, train_test_validation_split
from helpers.utils import remove_nan_rows, get_signal, portfolio_value, get_data_batch, get_lstm_input_output
from models import lstm_nn
from helpers.get_features import get_features, min_max_scaling


# other-params
np.set_printoptions(linewidth=75*3+5, edgeitems=6)
pl.rcParams.update({'font.size': 6})

# hyper-params
batch_size = 1024
learning_rate = 0.001
drop_keep_prob = 0.6
value_moving_average = 50
split = (0.7, 0.2, 0.1)
plotting = False
saving = False
time_steps = 5

# load data
oanda_data = np.load('data\\EUR_USD_H1.npy')  # [-50000:]
output_data_raw = price_to_binary_target(oanda_data, delta=0.00027)
price_data_raw = extract_timeseries_from_oanda_data(oanda_data, ['closeMid'])
input_data_raw, input_data_dummy_raw = get_features(oanda_data)
price_data_raw = np.concatenate([[[0]],
                                 (price_data_raw[1:] - price_data_raw[:-1]) / (price_data_raw[1:] + 1e-10)], axis=0)

# prepare data
input_data, output_data, input_data_dummy, price_data = \
    remove_nan_rows([input_data_raw, output_data_raw, input_data_dummy_raw, price_data_raw])
input_data_scaled_no_dummies = (input_data - min_max_scaling[1, :]) / (min_max_scaling[0, :] - min_max_scaling[1, :])
input_data = np.concatenate([input_data_scaled_no_dummies, input_data_dummy], axis=1)
input_data, output_data = get_lstm_input_output(input_data, output_data, time_steps=time_steps)
price_data = price_data[-len(input_data):]

# split to train,test and cross validation
input_train, input_test, input_cv, output_train, output_test, output_cv, price_train, price_test, price_cv = \
    train_test_validation_split([input_data, output_data, price_data], split=split)

# get dims
_, _, input_dim = np.shape(input_data)
_, output_dim = np.shape(output_data)

# forward-propagation
x, y, logits, y_, learning_r, drop_out = lstm_nn(input_dim, output_dim, time_steps=time_steps, n_hidden=[8])

# tf cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.AdamOptimizer(learning_r).minimize(cost)

# init session
cost_hist_train, cost_hist_test, value_hist_train, value_hist_test, value_hist_cv, value_hist_train_ma, \
    value_hist_test_ma, value_hist_cv_ma, step, step_hist, saving_score = [], [], [], [], [], [], [], [], 0, [], 0.05
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
while True:

    if step == 30000:
        break

    # train model
    x_train, y_train = get_data_batch([input_train, output_train], batch_size, sequential=False)
    _, cost_train = sess.run([train_step, cost],
                             feed_dict={x: x_train, y: y_train, learning_r: learning_rate, drop_out: drop_keep_prob})

    # keep track of stuff
    step += 1
    if step % 100 == 0 or step == 1:

        # get y_ predictions
        y_train_pred = sess.run(y_, feed_dict={x: input_train, drop_out: drop_keep_prob})
        cost_test, y_test_pred = sess.run([cost, y_], feed_dict={x: input_test, y: output_test, drop_out: drop_keep_prob})
        y_cv_pred = sess.run(y_, feed_dict={x: input_cv, y: output_cv, drop_out: drop_keep_prob})

        # get portfolio value
        signal_train, signal_test, signal_cv = get_signal(y_train_pred), get_signal(y_test_pred), get_signal(y_cv_pred)
        value_train = 1 + np.cumsum(np.sum(signal_train[:-1] * price_train[1:], axis=1))
        value_test = 1 + np.cumsum(np.sum(signal_test[:-1] * price_test[1:], axis=1))
        value_cv = 1 + np.cumsum(np.sum(signal_cv[:-1] * price_cv[1:], axis=1))

        # save history
        step_hist.append(step)
        cost_hist_train.append(cost_train)
        cost_hist_test.append(cost_test)
        value_hist_train.append(value_train[-1])
        value_hist_test.append(value_test[-1])
        value_hist_cv.append(value_cv[-1])
        value_hist_train_ma.append(np.mean(value_hist_train[-value_moving_average:]))
        value_hist_test_ma.append(np.mean(value_hist_test[-value_moving_average:]))
        value_hist_cv_ma.append(np.mean(value_hist_cv[-value_moving_average:]))

        print('Step {}: train {:.4f}, test {:.4f}'.format(step, cost_train, cost_test))

        if plotting:

            pl.figure(1, figsize=(3, 7), dpi=80, facecolor='w', edgecolor='k')

            pl.subplot(211)
            pl.title('cost function')
            pl.plot(step_hist, cost_hist_train, color='darkorange', linewidth=0.3)
            pl.plot(step_hist, cost_hist_test, color='dodgerblue', linewidth=0.3)

            pl.subplot(212)
            pl.title('Portfolio value')
            pl.plot(step_hist, value_hist_train, color='darkorange', linewidth=0.3)
            pl.plot(step_hist, value_hist_test, color='dodgerblue', linewidth=0.3)
            pl.plot(step_hist, value_hist_cv, color='magenta', linewidth=1)
            pl.plot(step_hist, value_hist_train_ma, color='tomato', linewidth=1.5)
            pl.plot(step_hist, value_hist_test_ma, color='royalblue', linewidth=1.5)
            pl.plot(step_hist, value_hist_cv_ma, color='black', linewidth=1.5)
            pl.pause(1e-10)

            # save if some complicated rules
        if saving:
            current_score = 0 if value_test[-1] < 0.01 or value_cv[-1] < 0.01 \
                else np.average([value_test[-1], value_cv[-1]])
            saving_score = current_score if saving_score < current_score else saving_score
            if saving_score == current_score and saving_score > 0.05:
                saver.save(sess, 'saved_models/lstm-v1-avg_score{:.3f}'.format(current_score), global_step=step)
                print('Model saved. Average score: {:.2f}'.format(current_score))

                pl.figure(2)
                pl.plot(value_test, linewidth=0.2)
                pl.plot(value_cv, linewidth=2)
                pl.pause(1e-10)


def save_plot():
    pl.figure(1, figsize=(3, 7), dpi=80, facecolor='w', edgecolor='k')

    pl.subplot(211)
    pl.title('cost function')
    pl.plot(step_hist, cost_hist_train, color='darkorange', linewidth=0.3)
    pl.plot(step_hist, cost_hist_test, color='dodgerblue', linewidth=0.3)

    pl.subplot(212)
    pl.title('Portfolio value')
    pl.plot(step_hist, value_hist_train, color='darkorange', linewidth=0.3)
    pl.plot(step_hist, value_hist_test, color='dodgerblue', linewidth=0.3)
    pl.plot(step_hist, value_hist_cv, color='magenta', linewidth=1)
    pl.plot(step_hist, value_hist_train_ma, color='tomato', linewidth=1.5)
    pl.plot(step_hist, value_hist_test_ma, color='royalblue', linewidth=1.5)
    pl.plot(step_hist, value_hist_cv_ma, color='black', linewidth=1.5)
    pl.pause(1e-10)

    pl.savefig('lstm_v1_{:.3f}_{:.3f}.png'.format(learning_rate, value_cv[-1]))
    pl.close()

save_plot()
