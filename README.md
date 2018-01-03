# Trading-Bot

Under development.

To do list:

1. **[Easy way to get historical data](helpers/get_historical_data.py)**

2. **[Set up code for managing orders (open, close, etc)](helpers/oanda_api_helpers.py)**

3. **[Engineer some features](helpers/get_features.py) and explore.**

   Most of the features are just raw indicators from [ta-lib](https://github.com/mrjbq7/ta-lib). I've hand picked them and made sure they    do not correlate too much. Additionally I've made a few dummy variables for market hours in major markets. 

   <p align="center"> 
      <img src="/images/feature_heatmap.png">
   </p>

3. **Train models:

   **Things to note**
   
   Results are terribly (!!!) dependent on randomization. My guess is, because the surface of objective function is very rough, each random initialization of weights and random pick of first training batches leads to new local optima. Therefore, to find a really good fit each model should be trained multiple times.
   
   Objective function is jumping up and down like crazy because batches of input are not homogenious (the set of 'rules' by which objective function is optimized changes dramatically from batch to batch). Nonetheless, it slowly moves towards some kind of optima (alright, not always).
   
   All the training is done using hourly data of EUR/USD. Once I'm comfortible with data exploration, models and stuff, I should try other pairs as well.

     **Try 1.** Predict the direction of price in the next time period. Target values [1, 0, 0] for up, [0, 1, 0] for down [0, 0, 1] for flat. Train by minimizing cross entropy of error.
     
   | [logistic regression](/train_logistic_regression_v1.py) | [lstm net](/train_lstm_v1.py) | [convolutional net](/train_cnn_v1.py) |
   | ------------------- | -------- | ----------------- |
   | <img src="/images/lr_v1_1.png"> | <img src="/images/lstm_v1_1.png"> | <img src="/images/lr_v1_1.png"> |
   
   <p align="center"> 
      <img src="/images/legend_one_fits_all.png">
   </p>
   
     **Try 2.** Instead of predicting price direction, allocate the funds to buy, sell, do not enter positions directly. For instance [0.5, 0.2, 0.3] would indicate to buy 0.5 units, sell 0.2 units and keep in cash 0.3 units. In this case target labels are not provided and the model is trained by maximizing objective function (hourly average return). 
   
   | [logistic regression](/train_logistic_regression_v2.py) | [lstm net](/train_lstm_v2.py) | [convolutional net](/train_cnn_v2.py) |
   | ------------------- | -------- | ----------------- |
   | <img src="/images/lr_v2_1.png"> | <img src="/images/lstm_v2_1.png"> | <img src="/images/lr_v2_1.png"> |
   
   <p align="center"> 
      <img src="/images/legend_one_fits_all.png">
   </p>
   
      **Try 3.** ...

**5. [Wrap things up and use the models to manage a single asset portfolio.](/main.py)**

   So far, logistic regression FTW!

