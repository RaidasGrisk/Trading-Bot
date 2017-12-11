"""
Features

https://github.com/mrjbq7/ta-lib
https://cryptotrader.org/talib
"""

import numpy as np
from talib.abstract import *
from utils import prep_data_for_feature_gen


def get_features(oanda_data):
    """Given OANDA data get some specified indicators using TA-Lib
    This is unfinished work. For now just random unstructured indicators
    """

    inputs = prep_data_for_feature_gen(oanda_data)

    # overlap studies
    par_sar = SAREXT(inputs)
    outm, outf = MAMA(inputs, optInFastLimit=12, optInSlowLimit=24)
    upper, middle, lower = BBANDS(inputs,
                                  optInTimePeriod=12,
                                  optInNbDevUp=2,
                                  optInNbDevDn=2,
                                  optinMAType='EMA')

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

    all_indicators = np.array([par_sar, outm, outf, upper, middle, lower, bop, cci, adx, cmo,
                               slowk, slowd, macd1, macd2, macd3, stocf1, stockf2, rsi1, rsi2,
                               ados, ht_sine1, ht_sine2, ht_phase, ht_trend, wcp, avg_range])

    return all_indicators.T  # transpose to get (data_points, features)

