"""
Features

https://github.com/mrjbq7/ta-lib
https://cryptotrader.org/talib
"""

import numpy as np
from datetime import datetime, timedelta
from talib.abstract import *
from helpers.utils import extract_timeseries_from_oanda_data


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

    all_indicators = np.array([price_change, volume_change, par_sar, outm, outf, upper, middle, lower, bop, cci, adx,
                               cmo, macd1, macd2, macd3, stocf1, stockf2, rsi1, rsi2,
                               ados, ht_sine1, ht_sine2, ht_phase, wcp, avg_range])

    all_dummies = np.array([ht_trend, mrkt_london, mrkt_ny, mrkt_sydney, mrkt_tokyo])

    return all_indicators.T, all_dummies.T  # transpose to get (data_points, features)
