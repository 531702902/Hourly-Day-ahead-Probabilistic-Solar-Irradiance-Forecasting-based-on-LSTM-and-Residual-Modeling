# Hourly-Day-ahead-Probabilistic-Solar-Irradiance-Forecasting-based-on-LSTM-and-Residual-Modeling
pre-data.rar contains the data downloaded from the midcdmz.nrel.gov/. Meanwhile,pre-data.rar has been normalized to [-1.1].


This project is mainly for others to reproduce the thesis work.
wx_LSTM_train.py and wx_LSTM_test.py are used to build LSTM residuals.
RenewResult.py is used to construct Laplacian and Gaussian prediction intervals with different confidence levels.

LSTMTrainerror.csv and LSTMTesterror.csv are the training set and test set error respectively.Laplace and Gaussian confidence intervals for different confidence levels can be constructed using LSTMTrainerror.csv, LSTMTesterror.csv, and RenewResult.py.
