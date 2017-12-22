# Trading-Bot

Under development.

To do list:
1. [Easy way to get historical data](helpers/get_historical_data.py)
2. [Set up code for managing orders (open, close, etc)](helpers/oanda_api_helpers.py)
3. [Engineer some features](helpers/get_features.py) and explore.

   Most of the features are just raw indicators from [ta-lib](https://github.com/mrjbq7/ta-lib). I've hand picked them and made sure they    do not correlate too much. Additionally I've made a few dummy variables for market hours in major markets. 

   <p align="center"> 
      <img src="/images/feature_heatmap.png">
   </p>

3. Train models:

   Try 1. Predict the direction of price in the next time period. Target values [1, 0, 0] for up, [0, 1, 0] for down [0, 0, 1] for flat. Train by minimizing cross enropy of error.
   
   Try 2. Instead of predicting price direction, allocate the funds to buy, sell, do not enter positions directly. For instance [0.5, 0.2, 0.3] would indicate to buy 0.5 units, sell 0.2 units and keep in cash 0.3 units. Train by directly maximizing profit.
   
   Try 3. ...


2. Wrap things up and use the models to manage a single asset portfolio.
