"""
Utils
"""

import numpy as np
import pandas as pd
import pylab as plt


def remove_nan_rows(items):
    """
    Get rid of rows if at least one value is nan in at least one item in input.
    Inputs a list of items to remove nans
    Returns arrays with filtered rows and unified length.
    """
    unified_mask = np.ones(len(items[0]), dtype=bool)
    for item in items:
        mask = np.any(np.isnan(item), axis=1)
        unified_mask[mask == True] = False
    return [item[unified_mask, :] for item in items]


def extract_timeseries_from_oanda_data(oanda_data, items):
    """Given keys of oanda data, put it's contents into an array"""
    output = []
    for item in items:
        time_series = np.array([x[item] for x in oanda_data]).reshape(len(oanda_data), 1)
        output.append(time_series)
    return output if len(output) > 1 else output[0]


def price_to_binary_target(oanda_data, delta=0.0001):
    """Quick and dirty way of constructing output where:
    [1, 0, 0] rise in price
    [0, 1, 0] price drop
    [0, 0, 1] no change (flat)
    """
    price = extract_timeseries_from_oanda_data(oanda_data, ['closeMid'])
    price_change = np.array([x1 / x2 - 1 for x1, x2 in zip(price[1:], price)])
    price_change = np.concatenate([[[0]], price_change])
    binary_price = np.zeros(shape=(len(price), 3))
    binary_price[-1] = np.nan
    for data_point in range(len(price_change) - 1):
        if price_change[data_point+1] > 0 and price_change[data_point+1] - delta > 0:  # price will drop
            column = 0
        elif price_change[data_point+1] < 0 and price_change[data_point+1] + delta < 0:  # price will rise
            column = 1
        else:  # price will not change
            column = 2
        binary_price[data_point][column] = 1

    # print target label distribution
    data_points = len(binary_price[:-1])
    print('Rise: {:.2f}, Drop: {:.2f}, Flat: {:.2f}'.format(np.sum(binary_price[:-1, 0]) / data_points,
                                                            np.sum(binary_price[:-1, 1]) / data_points,
                                                            np.sum(binary_price[:-1, 2]) / data_points))

    # print df to check if no look-ahead bias is introduced
    print(pd.DataFrame(np.concatenate([np.around(price, 5),
                                       np.around(price_change, 4),
                                       binary_price.astype(int)], axis=1)[:10, :]))

    return binary_price


def train_test_validation_split(list_of_items, split=(0.5, 0.35, 0.15)):
    """Splits data into train, test, validation samples"""
    train, test, cv = split
    id_train = int(len(list_of_items[0]) * train)
    id_test = int(len(list_of_items[0]) * (train + test))

    split_tuple = ()
    for item in list_of_items:
        train_split = item[:id_train]
        test_split = item[id_train:id_test]
        cv_split = item[id_test:]
        split_tuple = split_tuple + (train_split, test_split, cv_split)
    return split_tuple


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


def portfolio_value(price_change, signal, trans_costs=0.000):
    """Return portfolio value.
    IMPORTANT!
    signal received from last fully formed candle
    percentage price change over last fully formed candle and previous period"""
    # signal = signal_train
    # price_change = price_data
    signal_percent = signal[:-1] * price_change[1:]
    transaction_costs = np.zeros_like(signal_percent)
    for i in range(len(signal)-1):
        transaction_costs[i] = [trans_costs if signal[i] != signal[i+1] and signal[i+1] != 0 else 0]
    value = np.cumsum(signal_percent - transaction_costs) + 1
    # full = np.concatenate([signal, np.concatenate([[[0]], transaction_costs], axis=0)], axis=1)
    return value


def get_data_batch(list_of_items, batch_size, sequential):
    """Returns a batch of data. A batch of sequence or random points."""
    if sequential:
        indexes = np.random.randint(len(list_of_items[0]) - (batch_size+1))
    else:
        indexes = np.random.randint(0, len(list_of_items[0]), batch_size)
    batch_list = []
    for item in list_of_items:
        batch = item[indexes:indexes+batch_size, ...] if sequential else item[indexes, ...]
        batch_list.append(batch)
    return batch_list


def get_lstm_input_output(x, y, time_steps):
    """Returns sequential lstm shaped data [batch_size, time_steps, features]"""
    data_points, _ = np.shape(x)
    x_batch_reshaped = []
    for i in range(data_points - time_steps):
        x_batch_reshaped.append(x[i: i+time_steps, :])
    return np.array(x_batch_reshaped), y[time_steps:]


def get_cnn_input_output(x, y, time_steps=12):
    """Returns sequential cnn shaped data [batch_size, features, time_steps]"""
    data_points, _ = np.shape(x)
    x_batch_reshaped = []
    for i in range(data_points - time_steps):
        x_batch_reshaped.append(x[i:i+time_steps, :])
    x_batch_reshaped = np.transpose(np.array([x_batch_reshaped]), axes=(1, 3, 2, 0))
    return np.array(x_batch_reshaped), y[time_steps:]


def plot_roc_curve(y_pred_prob, y_target):
    """Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html"""
    from sklearn.metrics import roc_curve, auc
    # roc curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_target[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_target.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc["micro"], fpr["micro"], tpr["micro"]


def y_trans(y, to_1d):
    """Transform y data from [1, -1] to [[1, 0, 0], [0, 1, 0]] and vice versa"""
    if to_1d:
        y_flat = y.argmax(axis=1)
        map = {0: 1, 1: -1, 2: 0}
        y_new = np.copy(y_flat)
        for k, v in map.items():
            y_new[y_flat == k] = v
        return y_new.reshape(-1, 1)
    else:
        y_new = np.zeros(shape=(len(y), 3))
        for i in range(len(y_new)):
            index = [0 if y[i] == 1 else 1 if y[i] == -1 else 2][0]
            y_new[i, index] = 1
        return y_new


def min_max_scale(input_train, input_test, input_cv, std_dev_threshold=2.1):
    from sklearn.preprocessing import MinMaxScaler

    # get rid of outliers
    input_train_df = pd.DataFrame(input_train)
    input_train_no_outliers = input_train_df[input_train_df.apply(
        lambda x: np.abs(x - x.median()) / x.std() < std_dev_threshold).all(axis=1)].as_matrix()

    scaler = MinMaxScaler()
    scaler.fit(input_train_no_outliers)

    input_train_scaled = scaler.fit_transform(input_train)
    input_test_scaled = scaler.fit_transform(input_test)
    input_cv_scaled = scaler.fit_transform(input_cv)

    return input_train_scaled, input_test_scaled, input_cv_scaled


def get_pca(input_train, input_test, input_cv, threshold=0.01):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(input_train)
    plt.plot(pca.explained_variance_ratio_)
    nr_features = np.sum(pca.explained_variance_ratio_ > threshold)

    input_train_pca = pca.fit_transform(input_train)
    input_test_pca = pca.fit_transform(input_test)
    input_cv_pca = pca.fit_transform(input_cv)

    input_train_pca = input_train_pca[:, :nr_features]
    input_test_pca = input_test_pca[:, :nr_features]
    input_cv_pca = input_cv_pca[:, :nr_features]

    return input_train_pca, input_test_pca, input_cv_pca


def get_poloynomials(input_train, input_test, input_cv, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    poly.fit(input_train)

    input_train_poly = poly.fit_transform(input_train)
    input_test_poly = poly.fit_transform(input_test)
    input_cv_poly = poly.fit_transform(input_cv)

    return input_train_poly, input_test_poly, input_cv_poly