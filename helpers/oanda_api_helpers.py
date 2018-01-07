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


# TODO: make sure send_request checks if order is through on weekends and no order_book is created
class TradingSession:

    # initiate objects
    def __init__(self, accountID, access_token):
        self.accountID = accountID
        self.access_token = access_token
        self.api = oandapyV20.API(access_token=access_token, environment="practice")
        self.order_book = self.oanda_order_book()

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
        if units < 0:
            if self.order_book[instrument]['order_type'] is (not None and -1):
                print('Short: {} (holding)'.format(instrument))
                return 1
        elif units > 0:
            if self.order_book[instrument]['order_type'] is (not None and 1):
                print('Long: {} (holding)'.format(instrument))
                return 1
        else:
            print('Units specified: 0')
            return 1

        # define parameters, create and send a request
        mkt_order = MarketOrderRequest(instrument=instrument, units=units)
        r = orders.OrderCreate(self.accountID, data=mkt_order.data)
        request_data = self.send_request(r)

        # check if request was fulfilled and save its ID
        if request_data is not 1:
            instrument = request_data['orderCreateTransaction']['instrument']
            self.order_book[instrument]['tradeID'] = request_data['lastTransactionID']
            self.order_book[instrument]['order_type'] = -1 if units < 0 else 1
            print('{}: {}'.format('Long' if units > 0 else 'Short', instrument))
            return 0
        else:
            return 1

    def close_order(self, instrument):

        # check if position exist
        if self.order_book[instrument]['order_type'] is None:
            print('Position {} does not exist'.format(instrument))
            return 1

        # create and send a request
        r = trades.TradeClose(accountID=self.accountID, tradeID=self.order_book[instrument]['tradeID'])
        request_data = self.send_request(r)

        # check if request was fulfilled and clear it
        if request_data is not 1:
            instrument = request_data['orderCreateTransaction']['instrument']
            self.order_book[instrument]['order_type'] = None
            self.order_book[instrument]['tradeID'] = None
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

    def oanda_order_book(self):
        """Synchronize open positions with this object's order_book"""
        order_book_oanda = self.check_open_positions()
        order_book = {'EUR_USD': {'order_type': None, 'tradeID': None},
                      'AUD_JPY': {'order_type': None, 'tradeID': None}}
        for pos in order_book_oanda['positions']:
            try:
                trade_id = pos['long']['tradeIDs']
                order_type = 1
            except KeyError:
                trade_id = pos['short']['tradeIDs']
                order_type = -1
            order_book[pos['instrument']]['tradeID'] = trade_id
            order_book[pos['instrument']]['order_type'] = order_type
        return order_book

    def sync_with_oanda(self):
        self.order_book = self.oanda_order_book()

    def close_all_open_positions(self):
        """Close all opened positions"""

        # check oanda for open positions
        try:
            open_positions = self.check_open_positions()['positions'][0]
        except IndexError:
            self.order_book = self.oanda_order_book()
            print('No opened positions')
            return 0

        # get ID's of open positions
        trade_ids = []
        try:
            [trade_ids.append(x) for x in open_positions['short']['tradeIDs']]
        except KeyError:
            pass
        try:
            [trade_ids.append(x) for x in open_positions['long']['tradeIDs']]
        except KeyError:
            pass

        # close orders by ID
        [close_order_manually(self.accountID, self.access_token, x) for x in trade_ids]
        self.order_book = self.oanda_order_book()
        print('All positions closed')
        return 0


def close_order_manually(accountID, access_token, tradeID):
    """
    Closes order manually using tradeID.
    """
    api = oandapyV20.API(access_token=access_token, environment="practice")
    request = trades.TradeClose(accountID, tradeID)
    rv = api.request(request)
    print(json.dumps(rv, indent=2))
    return 0

