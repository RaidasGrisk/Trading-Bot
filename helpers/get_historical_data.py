import requests
import numpy as np
from datetime import datetime


def download_data_from_oanda(params):
    """
    Input: a dictionary with parameters
    http://developer.oanda.com/rest-live/rates/
    params = {'instrument': 'EUR_USD',
              'candleFormat': 'midpoint',
              'granularity': 'M15',
              'dailyAlignment': '0',
              'start': '2017-11-20',
              'count': '200'}

    Return: list of data up to the last available data point
    """

    # initiate variables
    data = []
    time_delta = None
    time_format = None
    finished = False

    while not finished:

        # get response
        response = requests.get(url='https://api-fxtrade.oanda.com/v1/candles', params=params).json()

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
