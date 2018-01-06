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

3. **Train models:**

   **Things to note**
   
   As a default, everything is done using 15 years of hourly data of EUR/USD. The dataset is split into 11 years of training data, 3 years of test data and 1.5 years of cross-validation data (keep this in mind when looking at portfolio value charts). Once I'm comfortible with data exploration, models and stuff, I should try other pairs as well. Or maybe a grand-model of huge-bunch-of-different-pairs at once!?

     **Try 1.** Predict the direction of price in the next time period. Target values [1, 0, 0] for up, [0, 1, 0] for down [0, 0, 1] for flat (sidenote: the threshold for minimum price change that is still considered flat is determined such that each label of up, down and flat makes roughly 1/3 of full dataset). Train by minimizing cross entropy of error.
     
   | [logistic regression](/train_logistic_regression_v1.py) | [lstm net](/train_lstm_v1.py) | [convolutional net](/train_cnn_v1.py) |
   | ------------------- | -------- | ----------------- |
   | <img src="/images/lr_v1.png"> | <img src="/images/lstm_v1.png"> | <img src="/images/cnn_v1.png"> |
   
   <p align="center"> 
      <img src="/images/legend_one_fits_all.png">
   </p>
   
     **Try 2.** Instead of predicting price direction, allocate the funds to buy, sell, do not enter positions directly. For instance [0.5, 0.2, 0.3] would indicate to buy 0.5 units, sell 0.2 units and keep in cash 0.3 units. In this case there are no target labels and the model is trained by maximizing objective function (hourly average return). 
   
   | [logistic regression](/train_logistic_regression_v2.py) | [lstm net](/train_lstm_v2.py) | [convolutional net](/train_cnn_v2.py) |
   | ------------------- | -------- | ----------------- |
   | <img src="/images/lr_v2_1.png"> | <img src="/images/lstm_v2_1.png"> | <img src="/images/cnn_v2_1.png"> |
   
   <p align="center"> 
      <img src="/images/legend_one_fits_all.png">
   </p>
   
    **Concusions after a few tries:**
      - LSTM's and CNN's have a big overfit (and at the same time underfit) problem that is hard to deal with. So, features should be picked very carefully for these models.
      - Learning rate does make a big difference. Training logistic regression with really small lr converges much better. It's probably a good idea to decrease lr again after a number of iterations.
      - Results are terribly (!!!) dependent on randomization. My guess is, because the surface of objective function is very rough, each random initialization of weights and random pick of first training batches leads to new local optima. Therefore, to find a really good fit each model should be trained multiple times.
      - Sometimes cost function is jumping up and down like crazy because batches of input are not homogenious (?) (the set of 'rules' by which objective function is optimized changes dramatically from batch to batch). Nonetheless, it slowly moves towards some kind of optima (not always! it might take a few tries of training from the beginning).
      - Adjusting hyper-parameters is hard and painful but might be worth the effort.
      - Training the models by optimizing cost cross-entropy (try 1) is much more effective then maximising return withought target labels (try 2).
   
      **Try 3.** ...

**5. [Wrap things up and use the models to manage a single asset portfolio.](/main.py)**

   So far, logistic regression FTW!
   
   **Ideas to try out later:**
   1. Use cnn to generate features, not predictions. Train these feeatures with logistic regression.
   2. Grand-model with multiple pairs as input and output.

