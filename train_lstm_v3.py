"""
Training lstm model v1
"""

import numpy as np
import pylab as pl
import tensorflow as tf
from helpers.utils import price_to_binary_target, min_max_scale, get_poloynomials, get_pca, train_test_validation_split
from helpers.utils import remove_nan_rows, get_signal, get_data_batch, get_lstm_input_output, plot_roc_curve
from models import lstm_nn
from helpers.get_features import get_features_v2


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
transaction_c = 0.000

# load data
oanda_data = np.load('data\\EUR_USD_H1.npy')[-50000:]
y_data = price_to_binary_target(oanda_data, delta=0.000275)
x_data = get_features_v2(oanda_data, time_periods=[10], return_numpy=False)

# separate, rearrange and remove nans
price = x_data['price'].as_matrix().reshape(-1, 1)
price_change = x_data['price_delta'].as_matrix().reshape(-1, 1)
x_data = x_data.drop(['price', 'price_delta'], axis=1).as_matrix()
price, price_change, x_data, y_data = remove_nan_rows([price, price_change, x_data, y_data])

# split to train,test and cross validation
input_train, input_test, input_cv, output_train, output_test, output_cv, price_train, price_test, price_cv = \
    train_test_validation_split([x_data, y_data, price_change], split=split)

# pre-process data: scale, pca, polynomial
input_train, input_test, input_cv = min_max_scale(input_train, input_test, input_cv, std_dev_threshold=2.5)
input_train, input_test, input_cv = get_pca(input_train, input_test, input_cv, threshold=0.01)
# input_train, input_test, input_cv = get_poloynomials(input_train, input_test, input_cv, degree=2)

# prep lstm format
input_train, output_train = get_lstm_input_output(input_train, output_train, time_steps=time_steps)
input_test, output_test = get_lstm_input_output(input_test, output_test, time_steps=time_steps)
input_cv, output_cv = get_lstm_input_output(input_cv, output_cv, time_steps=time_steps)

price_train = price_train[-len(input_train):]
price_test = price_test[-len(input_test):]
price_cv = price_cv[-len(input_cv):]

# get dims
_, _, input_dim = np.shape(input_train)
_, output_dim = np.shape(output_train)

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


# roc curve
roc_auc_train, fpr_train, tpr_train = plot_roc_curve(y_train_pred, output_train)
roc_auc_test, fpr_test, tpr_test = plot_roc_curve(y_test_pred, output_test)
roc_auc_cv, fpr_cv, tpr_cv = plot_roc_curve(y_cv_pred, output_cv)

plt.figure(2, figsize=(3, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Train area: {:0.2f}'.format(roc_auc_train))
plt.plot(fpr_test, tpr_test, color='dodgerblue', lw=2, label='Test area: {:0.2f}'.format(roc_auc_test))
plt.plot(fpr_cv, tpr_cv, color='magenta', lw=2, label='CV area: {:0.2f}'.format(roc_auc_cv))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
