"""
Features

https://github.com/mrjbq7/ta-lib
https://cryptotrader.org/talib
"""

import pandas as pd
import numpy as np
from datetime import datetime
from talib.abstract import *
import talib
from helpers.utils import extract_timeseries_from_oanda_data
import pylab as plt


def scale_data(input_data_no_dummies, split):
    """Scale NON DUMMY data given train, test, cv split"""
    from sklearn.preprocessing import MinMaxScaler
    train_split = int(len(input_data_no_dummies)*split[0])
    scaler = MinMaxScaler()
    scaler.fit(input_data_no_dummies[:train_split])
    return scaler.transform(input_data_no_dummies)


def prep_data_for_feature_gen(data):
    """Restructure OANDA data to use it for TA-Lib feature generation"""
    inputs = {
        'open': np.array([x['openMid'] for x in data]),
        'high': np.array([x['highMid'] for x in data]),
        'low': np.array([x['lowMid'] for x in data]),
        'close': np.array([x['closeMid'] for x in data]),
        'volume': np.array([float(x['volume']) for x in data])}
    return inputs


def get_features(oanda_data):
    """Given OANDA data get some specified indicators using TA-Lib
    This is unfinished work. For now just random unstructured indicators
    """

    # price and volume
    price, volume = extract_timeseries_from_oanda_data(oanda_data, ['closeMid', 'volume'])
    price_change = np.array([float(i) / float(j) - 1 for i, j in zip(price[1:], price)])
    volume_change = np.array([float(i) / float(j) - 1 for i, j in zip(volume[1:], volume)])
    price_change = np.concatenate([[np.nan], price_change], axis=0)
    volume_change = np.concatenate([[np.nan], volume_change], axis=0)

    inputs = prep_data_for_feature_gen(oanda_data)

    # overlap studies
    par_sar = SAREXT(inputs)
    outm, outf = MAMA(inputs, optInFastLimit=12, optInSlowLimit=24)
    upper, middle, lower = BBANDS(inputs,
                                  optInTimePeriod=12,
                                  optInNbDevUp=2,
                                  optInNbDevDn=2,
                                  optinMAType='EMA')
    upper = upper - price.ravel()
    middle = middle - price.ravel()
    lower = price.ravel() - lower

    # momentum
    bop = BOP(inputs)
    cci = CCI(inputs)
    adx = ADX(inputs, optInTimePeriod=24)
    cmo = CMO(inputs, optInTimePeriod=6)
    will = WILLR(inputs, optInTimePeriod=16)
    slowk, slowd = STOCH(inputs,
                         optInFastK_Period=5,
                         optInSlowK_Period=3,
                         optInSlowK_MAType=0,
                         optInSlowD_Period=3,
                         optInSlowD_MAType=0)
    macd1, macd2, macd3 = MACD(inputs,
                               optInFastPeriod=12,
                               optInSlowPeriod=6,
                               optInSignalPeriod=3)
    stocf1, stockf2 = STOCHF(inputs,
                             optInFastK_Period=12,
                             optInFastD_Period=6,
                             optInFastD_MAType='EXP')
    rsi1, rsi2 = STOCHRSI(inputs,
                          optInTimePeriod=24,
                          optInFastK_Period=12,
                          optInFastD_Period=24,
                          optInFastD_MAType='EXP')

    # volume indicators
    ados = ADOSC(inputs, optInFastPeriod=24, optInSlowPeriod=12)

    # cycle indicators
    ht_sine1, ht_sine2 = HT_SINE(inputs)
    ht_phase = HT_DCPHASE(inputs)
    ht_trend = HT_TRENDMODE(inputs)

    # price transform indicators
    wcp = WCLPRICE(inputs)

    # volatility indicators
    avg_range = NATR(inputs, optInTimePeriod=6)

    # TODO: pattern recognition (is this bullshit?)
    # pattern_indicators = []
    # for func in talib.get_function_groups()['Pattern Recognition']:
    #     result = eval(func + '(inputs)') / 100
    #     pattern_indicators.append(result)
    # pattern_indicators = np.array(pattern_indicators)

    # markets dummies
    time = np.array([datetime.strptime(x['time'], '%Y-%m-%dT%H:%M:%S.000000Z') for x in oanda_data])
    mrkt_london = [3 <= x.hour <= 11 for x in time]
    mrkt_ny = [8 <= x.hour <= 16 for x in time]
    mrkt_sydney = [17 <= x.hour <= 24 or 0 <= x.hour <= 1 for x in time]
    mrkt_tokyo = [19 <= x.hour <= 24 or 0 <= x.hour <= 3 for x in time]

    # sorting indicators
    all_indicators = np.array([price_change, volume_change, par_sar, outm, outf, upper, middle, lower, bop, cci, adx,
                               cmo, macd1, macd2, macd3, stocf1, stockf2, rsi1, rsi2,
                               ados, ht_sine1, ht_sine2, ht_phase, wcp, avg_range])

    all_dummies = np.array([ht_trend, mrkt_london, mrkt_ny, mrkt_sydney, mrkt_tokyo])

    return all_indicators.T, all_dummies.T  # transpose to get (data_points, features)


def get_features_v2(oanda_data, time_periods, return_numpy):
    """Returns all (mostly) indicators from ta-lib library for given time periods"""

    # load primary data
    inputs = prep_data_for_feature_gen(oanda_data)

    # get name of all the functions
    function_groups = ['Cycle Indicators',
                       'Momentum Indicators',
                       'Overlap Studies',
                       'Volume Indicators',
                       'Volatility Indicators',
                       'Statistic Functions']
    function_list = [talib.get_function_groups()[group] for group in function_groups]
    function_list = [item for sublist in function_list for item in sublist]  # flatten the list
    function_list.remove('MAVP')

    # price and volume
    price, volume = extract_timeseries_from_oanda_data(oanda_data, ['closeMid', 'volume'])
    price_change = np.array([float(i) / float(j) - 1 for i, j in zip(price[1:], price)])
    volume_change = np.array([float(i) / float(j) - 1 for i, j in zip(volume[1:], volume)])
    price_change = np.concatenate([[0], price_change], axis=0)
    volume_change = np.concatenate([[0], volume_change], axis=0)

    # get all indicators
    df_indicators = pd.DataFrame()
    df_indicators['price'] = price.ravel()
    df_indicators['price_delta'] = price_change
    df_indicators['volume_change'] = volume_change
    for func in function_list:
        if 'timeperiod' in getattr(talib.abstract, func).info['parameters']:
            for time_period in time_periods:
                indicator = getattr(talib.abstract, func)(inputs, timeperiod=time_period)
                if any(isinstance(item, np.ndarray) for item in indicator):  # if indicator returns > 1 time-series
                    indicator_id = 0
                    for x in indicator:
                        df_indicators[func + '_' + str(indicator_id) + '_tp_' + str(time_period)] = x
                        indicator_id += 1
                else:  # if indicator returns 1 time-series
                    df_indicators[func + '_tp_' + str(time_period)] = indicator
        else:
            indicator = getattr(talib.abstract, func)(inputs)
            if any(isinstance(item, np.ndarray) for item in indicator):
                indicator_id = 0
                for x in indicator:
                    df_indicators[func + str(indicator_id)] = x
                    indicator_id += 1
            else:
                df_indicators[func] = indicator

    # manual handling of features
    df_indicators['AD'] = df_indicators['AD'].pct_change()
    df_indicators['OBV'] = df_indicators['OBV'].pct_change()
    df_indicators['HT_DCPERIOD'] = (df_indicators['HT_DCPERIOD'] > pd.rolling_mean(df_indicators['HT_DCPERIOD'], 50)).astype(float)
    df_indicators['HT_DCPHASE'] = (df_indicators['HT_DCPHASE'] > pd.rolling_mean(df_indicators['HT_DCPHASE'], 10)).astype(float)
    df_indicators['ADX_tp_10'] = (df_indicators['ADX_tp_10'] > pd.rolling_mean(df_indicators['ADX_tp_10'], 10)).astype(float)
    df_indicators['MACD0'] = df_indicators['MACD0'] - df_indicators['MACD1']
    df_indicators['MINUS_DI_tp_10'] = (df_indicators['MINUS_DI_tp_10'] > pd.rolling_mean(df_indicators['MINUS_DI_tp_10'], 20)).astype(float)
    df_indicators['RSI_tp_10'] = (df_indicators['RSI_tp_10'] > pd.rolling_mean(df_indicators['RSI_tp_10'], 15)).astype(float)
    df_indicators['ULTOSC'] = (df_indicators['ULTOSC'] > pd.rolling_mean(df_indicators['ULTOSC'], 15)).astype(float)
    df_indicators['BBANDS_0_tp_10'] = df_indicators['BBANDS_0_tp_10'] - df_indicators['price']
    df_indicators['BBANDS_1_tp_10'] = df_indicators['BBANDS_1_tp_10'] - df_indicators['price']
    df_indicators['BBANDS_2_tp_10'] = df_indicators['BBANDS_2_tp_10'] - df_indicators['price']
    df_indicators['DEMA_tp_10'] = df_indicators['DEMA_tp_10'] - df_indicators['price']
    df_indicators['EMA_tp_10'] = df_indicators['EMA_tp_10'] - df_indicators['price']
    df_indicators['HT_TRENDLINE'] = df_indicators['HT_TRENDLINE'] - df_indicators['price']
    df_indicators['KAMA_tp_10'] = df_indicators['KAMA_tp_10'] - df_indicators['price']
    df_indicators['MAMA0'] = df_indicators['MAMA0'] - df_indicators['price']
    df_indicators['MAMA1'] = df_indicators['MAMA1'] - df_indicators['price']
    df_indicators['MIDPOINT_tp_10'] = df_indicators['MIDPOINT_tp_10'] - df_indicators['price']
    df_indicators['MIDPRICE_tp_10'] = df_indicators['MIDPRICE_tp_10'] - df_indicators['price']
    df_indicators['SMA_tp_10'] = df_indicators['SMA_tp_10'] - df_indicators['price']
    df_indicators['T3_tp_10'] = df_indicators['T3_tp_10'] - df_indicators['price']
    df_indicators['TEMA_tp_10'] = df_indicators['TEMA_tp_10'] - df_indicators['price']
    df_indicators['TRIMA_tp_10'] = df_indicators['TRIMA_tp_10'] - df_indicators['price']
    df_indicators['WMA_tp_10'] = df_indicators['WMA_tp_10'] - df_indicators['price']
    df_indicators['SAR'] = df_indicators['SAR'] - df_indicators['price']
    df_indicators['LINEARREG_tp_10'] = df_indicators['LINEARREG_tp_10'] - df_indicators['price']
    df_indicators['LINEARREG_INTERCEPT_tp_10'] = df_indicators['LINEARREG_INTERCEPT_tp_10'] - df_indicators['price']
    df_indicators['TSF_tp_10'] = df_indicators['TSF_tp_10'] - df_indicators['price']

    # markets dummies
    time = np.array([datetime.strptime(x['time'], '%Y-%m-%dT%H:%M:%S.000000Z') for x in oanda_data])
    df_indicators['mrkt_london'] = np.array([3 <= x.hour <= 11 for x in time]).astype(int)
    df_indicators['mrkt_ny'] = np.array([8 <= x.hour <= 16 for x in time]).astype(int)
    df_indicators['mrkt_sydney'] = np.array([17 <= x.hour <= 24 or 0 <= x.hour <= 1 for x in time]).astype(int)
    df_indicators['mrkt_tokyo'] = np.array([19 <= x.hour <= 24 or 0 <= x.hour <= 3 for x in time]).astype(int)

    print('Features shape: {}'.format(df_indicators.shape))

    return df_indicators.as_matrix() if return_numpy else df_indicators


# # min max scaling params (needs to be created manually (for now) or better use scilearn min_max scaler
# # min max parameters for scaling
# import pandas as pd
# oanda_data = np.load('data\\AUD_JPY_H1.npy')[-50000:]
# all_indicators, all_dummies = get_features(oanda_data)
# length = int(len(all_indicators) * 0.5)
# all_indicators = pd.DataFrame(all_indicators[:length, ])
# all_indicators_pd = all_indicators[all_indicators.apply(lambda x: np.abs(x - x.median()) / x.std() < 3).all(axis=1)]
# all_indicators_np = all_indicators_pd.as_matrix()
#
# min_max_parameters = np.array([np.nanmax(all_indicators_np[:, :length].T, axis=1),
#                                np.nanmin(all_indicators_np[:, :length].T, axis=1)])

# eur usd
min_max_scaling = np.array([[1.86410584e-03,   2.01841085e+00,   1.19412800e+00,
                             1.19447352e+00,   1.19295244e+00,   2.70961491e-03,
                             1.32700000e-03,   4.05070743e-03,   9.86577181e-01,
                             2.51521519e+02,   3.64593211e+01,   5.84544775e+01,
                             1.52468944e-03,   1.44255282e-03,   3.38887291e-04,
                             9.91166078e+01,   9.54336553e+01,   1.00000000e+02,
                             1.00000000e+02,   7.34727536e+03,   2.47949723e-01,
                            -5.09698958e-01,   2.06138115e+02,   1.19570250e+00,
                             1.04819528e-01],
                           [-1.30372596e-03,  -6.84790089e-01,  -1.19592000e+00,
                             1.18566979e+00,   1.15180427e+00,   2.90481254e-05,
                            -1.93000000e-03,   3.75062541e-05,  -9.53846154e-01,
                            -2.35424245e+02,   1.29761986e+01,  -2.20316967e+01,
                             8.84276800e-05,   1.76158803e-04,  -2.39463856e-04,
                             3.67647059e-01,   9.65742018e+00,   0.00000000e+00,
                            -2.96059473e-15,  -1.71810219e+03,  -4.40536468e-01,
                            -9.46300629e-01,   1.65643780e+02,   1.18376500e+00,
                             6.95891066e-02]])

