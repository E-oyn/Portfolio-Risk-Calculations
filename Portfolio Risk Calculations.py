from scipy.stats import norm
import pandas as pd
from pandas_datareader import data as wb
import numpy as np
import matplotlib.pyplot as plt
ticker = ['GOOG', 'FB', 'TSLA', 'MSFT', 'SAP']

df = pd.DataFrame()
for t in ticker:
    df[t] = wb.DataReader(t, data_source='yahoo', start='2020-1-1')['Adj Close']

#df.to_csv(r'C:\Users\Ege\Dropbox\WiSo 21-22\Python Work\Stock Calculations\stock_data.csv')

#Calculating Simple Returns

(df / df.iloc[0] * 100).plot(figsize = (15, 6));
#plt.show()

returns = (df / df.shift(1)) - 1
#A manual check for return calculation in case of an error
#print(returns.head())

#Assining the Portoflio Weights
weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
#Annual returns multiplied with 250 as the working day in a year
annual_returns = returns.mean() * 250
#Creating a Matrix for multipliying the yearly returns and their weight
weighted_return = np.dot(annual_returns, weights)
portfolio_return = str(round(weighted_return,2)*100) + ' %'

#print(portfolio_return)

#Calculating Portoflo Risk (Volatility)
returns_mean = returns[['GOOG', 'FB', 'TSLA', 'MSFT', 'SAP']].mean()*250
returns_std = returns[['GOOG', 'FB', 'TSLA', 'MSFT', 'SAP']].std() * 250 ** 5

#Creatign Covariance Matrix to check portfolios balancing
cov_matrix = returns.cov()*250
corr_matrix = returns.corr

pfolio_var = np.dot(weights.T, np.dot(returns.cov() * 250, weights))

pfolio_vol = (np.dot(weights.T, np.dot(returns.cov() * 250, weights))) ** 0.5

portfolio_volatility = (str(round(pfolio_vol, 2) * 100) + ' %')
#print(portfolio_volatility)

#Individual Stocks Variances

goog_var = returns['GOOG'].var()*250
fb_var = returns['FB'].var()*250
tsla_var = returns['TSLA'].var()*250
msft_var = returns['MSFT'].var()*250
sap_var = returns['SAP'].var()*250

#Portfolio Variance
portfolio_var = np.dot(weights.T, np.dot(returns.cov()*250,weights))
#Calculating Diversifiable Risk
dr = portfolio_var - (0.20**2*goog_var) - (0.20**2*fb_var) - (0.20**2*tsla_var) - (0.20**2*msft_var) - (0.20**2*sap_var)
diversifiable_risk =  (str(round(dr*100, 2)) + ' %')
#Calculating Non-Diversifiable Risk
non_divers = pfolio_var - dr


#Calculating Diversifiable Risk










#Now we compare this return to the indicies
i_tickers = ['^DJI', '^GSPC', '^IXIC']

ind_data = pd.DataFrame()

for inx in i_tickers:
    ind_data[inx] = wb.DataReader(inx, data_source='yahoo', start='2020-1-1')['Adj Close']
#print(ind_data.head())
(ind_data / ind_data.iloc[0] * 100).plot(figsize=(15, 6));
ind_returns = (ind_data / ind_data.shift(1)) - 1
annual_ind_returns = ind_returns.mean() * 250
