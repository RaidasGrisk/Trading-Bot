---------
### Table of Contents  

 1. [Intro](#Intro) 
 2. [Trading tools and helpers](#Tools)  
 3. [Training models v1](#train_v1) 
 4. [Training models v2](#train_v2) 
 5. [Final conclusions and ideas](#Concusion) 
 
<a name="Intro"/>

---------
### 1. Intro

This is a repo where I store code for training and making an automated FX trading bot.  
   
Essentially most of the work done here is about trying to train an accurate price movement classification model. But it as well contains all of the other necessary stuff like downloading historical or recent FX data and live-managing a demo trading account using OANDA's API.  

As a default, the training is done using 15 years of hourly data of EUR/USD. The dataset is split into 11 years of training data, 3 years of test data and 1.5 years of cross-validation data (keep this in mind when looking at portfolio value charts). MFeature engineering is mostly done using indicators from [ta-lib](https://github.com/mrjbq7/ta-lib) package.

Once I'm comfortible with data exploration, models and stuff, I should try other pairs as well. Or maybe a grand-model of huge-bunch-of-different-pairs at once!?

<a name="Tools"/>

---------
### 2. Trading tools and helpers


   * **[Easy way to get historical data.](helpers/get_historical_data.py)** Simple code to download historical (or current) data of selected instruments using OANDA's API.

   * **[Live trading portfolio manager.](helpers/oanda_api_helpers.py)** A class to manage real-time trading, open and close positions etc. using OANDA's API wrapper.
   
   * **[Kind of ready to use trading bot.](/main.py)** Final script combining everything to live manage a trading portfolio.
   
<a name="train_v1"/>

---------
### 3. Training models V1

First try is a bunch of 'quick and dirty' models with just a few features and some optimization experimentation. I've hand picked a few financial indicators and made sure they do not correlate too much. Additionally I've made a few dummy variables for market hours in major markets.  

   <p align="center"> 
      <img src="/images/feature_heatmap.png">
   </p>


**Predicting price direction.** 

Predict the direction of price in the next time period. Target values [1, 0, 0] for up, [0, 1, 0] for down [0, 0, 1] for flat (sidenote: the threshold for minimum price change that is still considered flat is determined such that each label of up, down and flat makes roughly 1/3 of full dataset). Train by minimizing cross entropy of error.
     
   | [logistic regression](/train_logistic_regression_v1.py) | [lstm net](/train_lstm_v1.py) | [convolutional net](/train_cnn_v1.py) |
   | ------------------- | -------- | ----------------- |
   | <img src="/images/lr_v1.png"> | <img src="/images/lstm_v1.png"> | <img src="/images/cnn_v1.png"> |
   
   <p align="center"> 
      <img src="/images/legend_one_fits_all.png">
   </p>
   
**Predicting optimal positions allocation** 

Instead of predicting price direction, allocate the funds to buy, sell, do not enter positions directly. For instance [0.5, 0.2, 0.3] would indicate to buy 0.5 units, sell 0.2 units and keep in cash 0.3 units. In this case there are no target labels and the model is trained by maximizing objective function (hourly average return). 
   
   | [logistic regression](/train_logistic_regression_v2.py) | [lstm net](/train_lstm_v2.py) | [convolutional net](/train_cnn_v2.py) |
   | ------------------- | -------- | ----------------- |
   | <img src="/images/lr_v2.png"> | <img src="/images/lstm_v2.png"> | <img src="/images/cnn_v2.png"> |
   
   <p align="center"> 
      <img src="/images/legend_one_fits_all.png">       
   </p>
   
**Concusions v1:**
   - Optimizing by minimizing cost cross-entropy with target labels works (i.e. predicting price direction). Optimizing by maximizing average return without target labels does not work (i.e. predicting optimal positions allocation). Because of unstable / uneven gradients maybe..?
   - LSTM and CNN models suffer from overfit problem (and underfit as well) that is hard to deal with. So I'll have to filter out least important features if I want to make it work.
   - Learning rate does make a big difference. Training logistic regression with really small lr converges much better. It's probably a good idea to decrease lr again after a number of iterations.
   - Results are terribly (!!!) dependent on randomization. My guess is, because the surface of objective function is very rough, each random initialization of weights and random pick of first training batches leads to new local optima. Therefore, to find a really good fit each model should be trained multiple times.
   - Sometimes cost function is jumping up and down like crazy because batches of input are not homogenious (?) (the set of 'rules' by which objective function is optimized changes dramatically from batch to batch). Nonetheless, it slowly moves towards some kind of optima (not always! it might take a few tries of training from the beginning).
   - Adjusting hyper-parameters is hard but it seems it might be worth the effort.

<a name="train_v2"/>

---------
### 3. Training models V2

This time the idea was to: 
  1. Create dozens of features (ta-lib indicators) of varying periods. Roughly there is 80 indicators, some of which can vary in time-periods, so all-in-all it is reasonable to create ~250 features.
  2. Perform PCA to simplify everything and get rid of similar and unimportant highly correlated features.
  3. Experiment with polynomials.

**<p align="center"> Plot example of a few features after normalization </p>**
<p align="center"> <img src="/images/features_example.png"></p>

After trying multiple ways of combining the features polynomials and PCA, it seems that this approach did not increase the accuracy of the model. Just for future reference I unclude best ROC scores I was able to reach using this approach.

**<p align="center"> Receiver operating curve </p>**
<p align="center"> <img src="/images/ROC_lr_v2.png"></p>


**Conclusions v2:**

  1. Given only price and volume data, predicting price direction is not really accurate. 
  2. For predictions to be reasonable more features are needed. For instance sentiment data, other macroeconomic data or whatever.
  3. If not only possible profitable strategy would be, to use other models like position sizing and carefully entering trades to decrease total transaction costs.
  
  Here is an example of portfolio value given the best models. Unfortunately, the results change dramatically once transaction costs are accounted for.

**<p align="center"> Portfolio value w\ and w\o transaction costs </p>**

<p align="center"> <img src="/images/portfolio_value_1.png"> </p>

<a name="Concusion"/>

---------
### 5. Final remarks
   
   **Ideas to try out someday:**
   1. Use inner layers of cnn as features in logistic regression.
   2. Grand-model with multiple pairs as input and output.
   3. Use evolution strategies to optimize for stuff that has no smooth gradients: SL...





