"""
Utils
"""

import numpy as np


def remove_nan_rows(array):
    """
    Deletes rows if at least one value is nan.
    Returns array with filtered rows.
    """
    mask = np.any(np.isnan(array), axis=1)
    return array[~mask, :]


def extract_timeseries_from_oanda_data(oanda_data, time_series):
    """Give a key from oanda data and put it's contents into an array"""
    return np.array([x[time_series]for x in oanda_data]).reshape(len(oanda_data), 1)


def prep_data_for_feature_gen(data):
    """Restructure OANDA data to use it for TA-Lib feature generation"""
    inputs = {
        'open': np.array([x['openMid'] for x in data]),
        'high': np.array([x['highMid'] for x in data]),
        'low': np.array([x['lowMid'] for x in data]),
        'close': np.array([x['closeMid'] for x in data]),
        'volume': np.array([float(x['volume']) for x in data])}
    return inputs


def price_to_binary_target(oanda_data, delta=0.001):
    """Quick and dirty way of constructing output where:
    [1, 0, 0] rise in price
    [0, 1, 0] price drop
    [0, 0, 1] no change (flat)
    """
    price = extract_timeseries_from_oanda_data(oanda_data, 'closeMid')
    price_change = np.array([x1 / x2 - 1 for x1, x2 in zip(price[1:], price)])
    binary_price = np.zeros(shape=(len(price), 3))
    binary_price[-1] = np.nan
    for data_point in range(len(price_change)-1):
        if price_change[data_point] > 0 and price_change[data_point] - delta > 0:  # price will drop
            column = 0
        elif price_change[data_point] < 0 and price_change[data_point] + delta < 0:  # price will rise
            column = 1
        else:  # price will not change
            column = 2
        binary_price[data_point][column] = 1

    data_points = len(binary_price[:-1])
    print('Rise: {:.2f}, Drop: {:.2f}, Flat: {:.2f}'.format(np.sum(binary_price[:-1, 0]) / data_points,
                                                            np.sum(binary_price[:-1, 1]) / data_points,
                                                            np.sum(binary_price[:-1, 2]) / data_points))
    return binary_price


def get_signal(softmax_output):
    """Return an array of signals given softmax output"""
    signal_index = np.argmax(softmax_output, axis=1)
    signal = np.zeros(shape=(len(signal_index), 1))
    for index, point in zip(signal_index, range(len(signal))):
        if index == 0:
            signal[point] = 1
        elif index == 1:
            signal[point] = -1
        else:
            signal[point] = 0
    return signal


def portfolio_value(price, signal, trans_costs=0.0001):
    """Return portfolio value given price of an instrument, it's transaction costs and signal values"""
    price_change = np.array([x1 / x2 - 1 for x1, x2 in zip(price[1:], price)])
    signal_percent = signal[:-1] * price_change.reshape(len(price_change), 1)
    transaction_costs = np.zeros_like(signal_percent)
    for i in range(len(signal)-1):
        transaction_costs[i] = [trans_costs * price[i] if signal[i] != signal[i+1] else 0]
    value = np.cumsum(signal_percent - transaction_costs) + 1
    return value


def get_data_batch(x, y, batch_size):
    """Returns a batch of sequential data"""
    indexes = np.random.choice(len(y) - (batch_size+1))
    x_batch = x[indexes:indexes+batch_size, ...]
    y_batch = y[indexes:indexes+batch_size, ...]
    return x_batch, y_batch


def get_lstm_input_output(x, y, time_steps):
    """Returns a batch of sequential data for lst shaped like [batch_size, time_steps, features]"""
    data_points, _ = np.shape(x)
    x_batch_reshaped = []
    for i in range(data_points - time_steps):
        x_batch_reshaped.append(x[i: i+time_steps, :])
    return np.array(x_batch_reshaped), y[time_steps:]
