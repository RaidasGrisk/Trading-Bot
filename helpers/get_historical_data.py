import requests
import numpy as np
from datetime import datetime
import os


def download_data_from_oanda(params):
    """
    Input: a dictionary with parameters
    http://developer.oanda.com/rest-live/rates/
    params = {'instrument': 'EUR_USD',
              'candleFormat': 'midpoint',
              'granularity': 'M15',
              'dailyAlignment': '0',
              'start': '2017-11-20',
              'count': '5000'}

    Return: list of data up to the last available data point
    """

    # initiate variables
    data = []
    time_delta = None
    time_format = None
    finished = False

    while not finished:

        # get response
        try:
            response = requests.get(url='https://api-fxtrade.oanda.com/v1/candles', params=params).json()
        except ValueError:
            print('Something is wrong with oanda response')

        # append response
        data = np.append(data, np.array(response['candles']))

        # ascertain time delta (only once)
        if not time_delta:
            time_format = '%Y-%m-%dT%H:%M:%S.000000Z'
            time_delta = datetime.strptime(response['candles'][-1]['time'], time_format) - \
                         datetime.strptime(response['candles'][-2]['time'], time_format)

        # start from last time stamp
        params['start'] = (datetime.strptime(response['candles'][-1]['time'], time_format) + time_delta).isoformat('T')

        # check if finished
        finished = not response['candles'][-1]['complete']
        print('Done!') if finished else print(response['candles'][-1]['time'])

    return data


def download_multiple_instruments_and_save(instrument_list, params):
    """
    Downloads specified instruments and saves it to /data/'instrument'.npy

    instrument_list = ["AUD_JPY", "AUD_USD", "CHF_JPY",
                       "EUR_CAD", "EUR_CHF", "EUR_GBP",
                       "EUR_JPY", "EUR_USD", "GBP_CHF",
                       "GBP_JPY", "GBP_USD", "NZD_JPY",
                       "NZD_USD", "USD_CHF", "USD_JPY"]

    params = {'instrument': '',
              'candleFormat': 'midpoint',
              'granularity': 'M15',
              'dailyAlignment': '0',
              'start': '2017-11-20',
              'count': '5000'}

    Return: None it just saves the data
    """

    starting_time = params['start']
    for instrument in instrument_list:

        # download data
        params['instrument'] = instrument
        params['start'] = starting_time
        instrument_data = download_data_from_oanda(params)

        # save and track progress
        np.save('data\\{}_{}.npy'.format(instrument, params['granularity']), instrument_data)
        print('{} is finished!'.format(instrument))


def get_latest_oanda_data(instrument, granularity, count):
    """Returns last oanda data (with a length of count) for a given instrument and granularity"""

    params = {'instrument': instrument,
              'candleFormat': 'midpoint',
              'granularity': granularity,
              'dailyAlignment': '0',
              'count': count+1}  # +1 to make sure all returned candles are complete
    response = requests.get(url='https://api-fxtrade.oanda.com/v1/candles', params=params).json()
    data = np.array(response['candles'])

    # if last candle is complete, return full data (except first point), else omit last data point
    return data[1:] if data[-1]['complete'] else data[:-1]


# code to download a list of instruments
# download_multiple_instruments_and_save(instrument_list=["AUD_JPY", "AUD_USD", "CHF_JPY",
#                                                         "EUR_CAD", "EUR_CHF", "EUR_GBP",
#                                                         "EUR_JPY", "EUR_USD", "GBP_CHF",
#                                                         "GBP_JPY", "GBP_USD", "NZD_JPY",
#                                                         "NZD_USD", "USD_CHF", "USD_JPY"],
#                                        params={'instrument': '',
#                                                'candleFormat': 'midpoint',
#                                                'granularity': 'M1',
#                                                'dailyAlignment': '0',
#                                                'start': '2001-01-01',
#                                                'count': '5000'})
