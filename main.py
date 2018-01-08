"""
This is the main code for automated FX trading

"""

from apscheduler.schedulers.blocking import BlockingScheduler
from helpers.oanda_api_helpers import TradingSession
from helpers.utils import remove_nan_rows
from helpers.get_features import get_features, min_max_scaling
from helpers.get_historical_data import get_latest_oanda_data
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import time
import pytz


# some parameters
close_current_positions = True
start_on_spot = True

# oanda access keys
accountID = ''
access_token = ''
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

# other variables
log = pd.DataFrame()
tz = pytz.timezone('Europe/Vilnius')
start_time = str(datetime.datetime.now(tz))[:-13].replace(':', '-')
margin_rate = float(trading_sess.check_account_summary()['account']['marginRate'])
last_complete_candle_stamp = ''


def do_stuff_every_period():

    global log
    global start_time
    global last_complete_candle_stamp
    global margin_rate
    current_time = str(datetime.datetime.now(tz))[:-13]

    # estimate position size
    account_balance = np.around(float(trading_sess.check_account_summary()['account']['balance']), 0)
    funds_to_commit = account_balance * (1 / margin_rate)

    # download latest data
    # always check if new candle is present, because even after 5 seconds, it might be not formed if market is very calm
    # make sure this loop does not loop endlessly on weekends (this is configured in scheduler)
    while True:
        oanda_data = get_latest_oanda_data('EUR_USD', 'H1', 300)  # many data-points to increase EMA and such accuracy
        current_complete_candle_stamp = oanda_data[-1]['time']
        if current_complete_candle_stamp != last_complete_candle_stamp:  # if new candle is complete
            break
        time.sleep(5)
    last_complete_candle_stamp = current_complete_candle_stamp

    # get features
    input_data_raw, input_data_dummy = get_features(oanda_data)
    input_data, input_data_dummy = remove_nan_rows([input_data_raw, input_data_dummy])
    input_data_scaled_no_dummy = (input_data - min_max_scaling[1, :]) / (min_max_scaling[0, :] - min_max_scaling[1, :])
    input_data_scaled = np.concatenate([input_data_scaled_no_dummy, input_data_dummy], axis=1)

    # estimate signal
    y_pred = sess.run(y_, feed_dict={x: input_data_scaled[-1:, :], drop_out: 1})
    order_signal_id = y_pred.argmax()
    order_signal = [1, -1, 0][order_signal_id]  # 0 stands for buy, 1 for sell, 2 for hold

    # manage trading positions
    current_position = trading_sess.order_book['EUR_USD']['order_type']
    if current_position != order_signal:
        if current_position is not None:
            trading_sess.close_order('EUR_USD')
        trading_sess.open_order('EUR_USD', funds_to_commit * order_signal)
    else:
        print('{}: EUR_USD (holding)'.format(['Long', 'Short', 'Nothing'][order_signal_id]))

    # log
    new_log = pd.DataFrame([[current_time, oanda_data[-1]['closeMid'], y_pred]],
                           columns=['Datetime', 'Last input Price', 'y_pred'])
    log = log.append(new_log)
    log.to_csv('logs/log {}.csv'.format(start_time))

    print('{} | price: {:.5f} | signal: buy: {:.2f}, sell: {:.2f}, nothing: {:.2f}'
          .format(current_time, oanda_data[-1]['closeMid'], y_pred[0][0], y_pred[0][1], y_pred[0][2]))


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
