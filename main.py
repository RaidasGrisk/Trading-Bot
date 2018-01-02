"""
This is the main code for automated FX trading

"""

from helpers.oanda_api_helpers import TradingSession, close_order_manually
from helpers.utils import remove_nan_rows
from helpers.get_features import get_features, min_max_scaling
from helpers.get_historical_data import get_latest_oanda_data
import tensorflow as tf
import numpy as np
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# oanda access keys
accountID = '101-004-3943081-006'
access_token = 'fb12d7edd860927ce27467d8ec4aee94-1cb7ffc0e40d649b736315872a10c545'
model_name = 'lr-v2-avg_score0.204-64000'

# init trading session
trading_sess = TradingSession(accountID=accountID, access_token=access_token)

# init tf model
config = tf.ConfigProto(device_count={'GPU': 0})
sess = tf.Session(config=config)
saver = tf.train.import_meta_graph('saved_models/' + model_name + '.meta')
saver.restore(sess, tf.train.latest_checkpoint('saved_models/'))
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("Placeholder:0")
tf_op_to_restore = graph.get_tensor_by_name("Softmax:0")

# Do stuff every period
scheduler = BlockingScheduler()


@scheduler.scheduled_job(trigger='cron', day_of_week='0-6', hour='0-23', minute='0', second='5')
def do_stuff_every_period():

    # retrieve data and return signal
    oanda_data = get_latest_oanda_data('EUR_USD', 'H1', 64)
    input_data_raw, input_data_dummy = get_features(oanda_data)
    input_data, input_data_dummy = remove_nan_rows([input_data_raw, input_data_dummy])
    input_data_scaled_no_dummies = (input_data - min_max_scaling[1, :]) / (
            min_max_scaling[0, :] - min_max_scaling[1, :])
    input_data_scaled = np.concatenate([input_data_scaled_no_dummies, input_data_dummy], axis=1)
    y_ = sess.run(tf_op_to_restore, feed_dict={x: input_data_scaled})
    order_signal = y_.argmax()  # 0 stands for buy, 1 for sell, 2 for hold

    print('{} | signal: buy: {:.2f}, sell: {:.2f}, nothing: {:.2f}'.format(
        str(datetime.datetime.now())[:-4], y_[0][0], y_[0][1], y_[0][2]))

    # if signal long
    if order_signal == 0:
        if trading_sess.order_book['EUR_USD']['order_type'] == -1:
            trading_sess.close_order('EUR_USD')
        trading_sess.open_order('EUR_USD', 1)

    # if signal short
    elif order_signal == 1:
        if trading_sess.order_book['EUR_USD']['order_type'] == 1:
            trading_sess.close_order('EUR_USD')
        trading_sess.open_order('EUR_USD', -1)

    # else (uncharted waters)
    else:
        print('Do nothing')


# start
do_stuff_every_period()
scheduler.start()

# close_order_manually(accountID, access_token, 1579)
# trading_sess.check_open_positions()
# trading_sess.check_account_summary()
# trading_sess.order_book
