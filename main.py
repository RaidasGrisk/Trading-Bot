"""
This is the main code for automated FX trading

"""

from helpers.oanda_api_helpers import TradingSession, close_order_manually
from helpers.utils import remove_nan_rows
from helpers.get_features import get_features, min_max_scaling
from helpers.get_historical_data import get_latest_oanda_data
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# TODO: trading params like position size kelly criterion
position_size = 10000
close_current_positions = True
start_on_spot = False

# oanda access keys
accountID = '101-004-3943081-001'
access_token = 'fb12d7edd860927ce27467d8ec4aee94-1cb7ffc0e40d649b736315872a10c545'
model_name = 'lr-v1-avg_score1.454-2000'

# init trading session
trading_sess = TradingSession(accountID=accountID, access_token=access_token)
if close_current_positions:
    trading_sess.close_all_open_positions()

# init tf model
config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('saved_models/' + model_name + '.meta')
saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))
graph = tf.get_default_graph()
x = graph.get_tensor_by_name('Placeholder:0')
drop_out = graph.get_tensor_by_name('strided_slice_1:0')
y_ = graph.get_tensor_by_name('Softmax:0')

# global variables
log = pd.DataFrame()
start_time = str(datetime.datetime.now())[:-7].replace(':', '-')


def do_stuff_every_period():

    global log
    global start_time
    current_time = str(datetime.datetime.now())[:-7]

    # gather data and return signal
    oanda_data = get_latest_oanda_data('EUR_USD', 'H1', 64)
    input_data_raw, input_data_dummy = get_features(oanda_data)
    input_data, input_data_dummy = remove_nan_rows([input_data_raw, input_data_dummy])
    input_data_scaled_no_dummies = (input_data - min_max_scaling[1, :]) / (
            min_max_scaling[0, :] - min_max_scaling[1, :])
    input_data_scaled = np.concatenate([input_data_scaled_no_dummies, input_data_dummy], axis=1)

    # estimate signal
    y_pred = sess.run(y_, feed_dict={x: input_data_scaled, drop_out: 1})
    order_signal = y_pred.argmax()  # 0 stands for buy, 1 for sell, 2 for hold

    print('{} | price: {:.5f} | signal: buy: {:.2f}, sell: {:.2f}, nothing: {:.2f}'
          .format(current_time, oanda_data[-1]['closeMid'], y_pred[0][0], y_pred[0][1], y_pred[0][2]))

    # if signal long
    if order_signal == 0:
        if trading_sess.order_book['EUR_USD']['order_type'] == -1:
            trading_sess.close_order('EUR_USD')
        trading_sess.open_order('EUR_USD', position_size)

    # if signal short
    elif order_signal == 1:
        if trading_sess.order_book['EUR_USD']['order_type'] == 1:
            trading_sess.close_order('EUR_USD')
        trading_sess.open_order('EUR_USD', -position_size)

    # else (uncharted waters)
    else:
        print('Do nothing')

    # log
    new_log = pd.DataFrame([[current_time, oanda_data[-1]['closeMid'], y_pred]],
                           columns=['Datetime', 'Last input Price', 'y_pred'])
    log = log.append(new_log)
    log.to_csv('logs/log {}.csv'.format(start_time))


# Scheduler
scheduler = BlockingScheduler()
scheduler.add_job(do_stuff_every_period,
                  trigger='cron',
                  day_of_week='0-4',
                  hour='0-23',
                  minute='0',
                  second='5')

if start_on_spot:
    do_stuff_every_period()
scheduler.start()

# close_order_manually(accountID, access_token, 1603)
# trading_sess.check_open_positions()
# trading_sess.check_account_summary()
# trading_sess.order_book
