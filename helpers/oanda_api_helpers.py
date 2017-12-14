"""
This is where all the code related to interacting with oanda is stored.

Sources:
https://media.readthedocs.org/pdf/oanda-api-v20/latest/oanda-api-v20.pdf
https://github.com/hootnot/oanda-api-v20

"""

import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.positions as positions
from oandapyV20.contrib.requests import MarketOrderRequest
import json


class TradingSession:

    # initiate objects
    def __init__(self, accountID, access_token):
        self.accountID = accountID
        self.api = oandapyV20.API(access_token=access_token, environment="practice")
        self.opened_orders_list = {'EUR_USD': None,
                                   'AUD_JPY': None}

    # initiate methods
    def send_request(self, request):
        """
        Sends request to oanda API.
        Returns oanda's response if success, 1 if error.
        """
        try:
            rv = self.api.request(request)
            # print(json.dumps(rv, indent=2))
            return rv
        except oandapyV20.exceptions.V20Error as err:
            print(request.status_code, err)
            return 1

    def open_order(self, instrument, units):

        # check if position is already open
        if self.opened_orders_list[instrument] is not None:
            print('Position {} is already opened'.format(instrument))
            return 1

        # define parameters, create and send a request
        mkt_order = MarketOrderRequest(instrument=instrument, units=units)
        r = orders.OrderCreate(self.accountID, data=mkt_order.data)
        request_data = self.send_request(r)

        # check if request was fulfilled and save its ID
        if request_data is not 1:
            instrument = request_data['orderCreateTransaction']['instrument']
            self.opened_orders_list[instrument] = (request_data['lastTransactionID'])
            print('Opened: {}'.format(instrument))
            return 0
        else:
            return 1

    def close_order(self, instrument):

        # check if position exist
        if self.opened_orders_list[instrument] is None:
            print('Position {} does not exist'.format(instrument))
            return 1

        # create and send a request
        r = trades.TradeClose(accountID=self.accountID, tradeID=self.opened_orders_list[instrument])
        request_data = self.send_request(r)

        # check if request was fulfilled and clear it
        if request_data is not 1:
            instrument = request_data['orderCreateTransaction']['instrument']
            self.opened_orders_list[instrument] = None
            print('Closed: {}'.format(instrument))
            return 0
        else:
            return 1

    def check_open_positions(self):
        r = positions.OpenPositions(self.accountID)
        return self.send_request(r)

    def check_account_summary(self):
        r = accounts.AccountSummary(self.accountID)
        return self.send_request(r)


def close_order_manually(accountID, access_token, tradeID):
    """
    Closes order manually using tradeID.
    """
    api = oandapyV20.API(access_token=access_token, environment="practice")
    request = trades.TradeClose(accountID, tradeID)
    rv = api.request(request)
    print(json.dumps(rv, indent=2))
    return 0

