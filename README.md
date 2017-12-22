# Trading-Bot

Under development.

To do list:
1. [Easy way to get historical data](helpers/get_historical_data.py)
2. [Set up code for managing orders (open, close, etc)](helpers/oanda_api_helpers.py)
3. [Engineer some features](helpers/get_features.py) and explore.

Most of the features are just raw indicators from [ta-lib](https://github.com/mrjbq7/ta-lib). I've hand picked them and made sure they do not correlate too much. Additionally I've made a few dummy variables for market hours in major markets. 

![](/images/figure_1.png) ... ![](/images/figure_1-1.png)


3. Train and compare a bunch of models:
   - logistic regression
   - vanilla neural net
   - lstm net
   - conv net + lstm ne
   - anything else..?
2. Wrap things up and use the models to manage a single asset portfolio.
