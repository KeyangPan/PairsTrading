import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import List, Tuple, Dict



class backtest_MRP:

    def __init__(self, stock_weight, window_train, window_test):

        def get_portfolio_value(weights_single_stocks, data_window):
            data_window = data_window[weights_single_stocks.index] # rearange columns
            portfolio_value = (data_window * np.array(weights_single_stocks)).sum(axis=1) # daily portfolio value
            return portfolio_value
        
        self.portfolio_value_train = get_portfolio_value(stock_weight, window_train)
        self.portfolio_value_test = get_portfolio_value(stock_weight, window_test)


    def run(self):
        mu = self.portfolio_value_train.mean()
        sigma = self.portfolio_value_train.std()
        thr = self._threshold_gaussian(sigma)

        residuals_train = self.portfolio_value_train - mu
        residuals_test = self.portfolio_value_test - mu 

        trade_table_train = self._backtest(self.portfolio_value_train, residuals_train, thr)
        trade_table_test = self._backtest(self.portfolio_value_test, residuals_test, thr)

        return trade_table_train, trade_table_test

    def _backtest(self, portfolio_values, residuals, thr):
        ## step1: form a trade table
        trade_table = pd.DataFrame({'portfolio_value': portfolio_values,
                                    'residuals': residuals,
                                    'threshold': thr,
                            },
              index = residuals.index)

        tra_signals = np.where(residuals >= thr, -1, np.where(residuals <= -thr, 1, 0))
        trade_table["trade_signal"] = tra_signals

        # 当residual符号出现反转时,生成unwind signal
        prev_sign = np.sign(trade_table['residuals']).shift(1)
        curr_sign = np.sign(trade_table['residuals'])
        trade_table['unwind_signal'] = (curr_sign != prev_sign).astype(int)
        trade_table.loc[trade_table.index[0], 'unwind_signal'] = 0   

        ## step2: backtest the given period
        portfolio_value = 1 # initial value to be 1
        portfolio_value_list = []
        position_list = []
        prev_price = None
        position = 0 # 0:position close; 1:long spread; 2: short spread

        for idx, row in trade_table.iterrows():

            # current position is closed
            if position == 0:
                if row['trade_signal'] != 0: # open a position if receive the trade signal
                    position = row["trade_signal"]
            
            # current position is long
            elif position == 1:
                curr_price = row['portfolio_value']
                ret = (curr_price - prev_price) / np.abs(prev_price)
                portfolio_value = portfolio_value * (ret + 1)

                if row["unwind_signal"] == 1:
                    position = 0
            
            # current position is short
            elif position == -1:
                curr_price = row['portfolio_value']
                ret = (prev_price - curr_price) / np.abs(prev_price)
                portfolio_value = portfolio_value * (ret + 1)

                if row["unwind_signal"] == 1:
                    position = 0
            
            else:
                raise "Position invalid"

            prev_price = row['portfolio_value']
            position_list.append(position)
            portfolio_value_list.append(portfolio_value)   
        
        trade_table['position'] = position_list
        trade_table['cum_ret'] = portfolio_value_list

        return trade_table




    def _threshold_gaussian(self, sigma: float) -> float:
        """
        Computes the optimal trading threshold for a Gaussian-distributed residual.

        Parameters:
        - sigma: Standard deviation of the residual (sqrt of sigma2 from params).

        Returns:
        - Optimal threshold value that maximizes expected trading profit.

        Logic:
        - Defines a negative profit function based on the Gaussian CDF and the threshold.
        - Minimizes this function within bounds [0, 3*sigma] to find the optimal threshold.
        """
        def _neg_profit(x: float) -> float:
            # Negative profit: -(probability of exceeding x) * x
            return -1.0 * (1.0 - stats.norm.cdf(x / sigma)) * x

        sol = minimize_scalar(_neg_profit, bounds=(0.0, 3.0 * sigma), method="bounded")
        return sol.x
    
