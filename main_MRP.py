import cvxpy as cp

# Prevent CVXPY from attempting to import these solvers
cp.settings.CVXOPT_ATTEMPTS = 0
cp.settings.GLPK_ATTEMPTS = 0
cp.settings.GLPK_MI_ATTEMPTS = 0


import math
from data_loader import Data_loader
from pair_filter import NPD_filter
from pair_filter import OLS_filter
import matplotlib.pyplot as plt
from backtest import PairTradingBacktester_TrainTest
from optimizers import MRP_optimizer_scipy
import pandas as pd
from backtest_MRP import backtest_MRP



import time
import multiprocessing as mp
import pickle
import os

import warnings
warnings.filterwarnings('ignore')



def pipeline_MRP(window_train, window_test):

    ## 1. Pair selection phase
    # NPD selection
    NPD = NPD_filter(window_train)
    pairs, distance = NPD.select_pairs(percentile=5, verbose=False)


    # OLS selection
    OLS = OLS_filter(window_train)
    pairs, params = OLS.select_pairs(pairs, verbose=False, confidence_level=0.01)


    # backtest and select top k pairs with top sharpe ratio on train data
    top_k = 50

    # bt = PairTradingBacktester_TrainTest(pairs[:300], params[:300], window_train, window_test) # for testing purpose
    bt = PairTradingBacktester_TrainTest(pairs, params, window_train, window_test) # complete
    _ = bt.run(top_k=top_k)
    spreads_train  = bt.train_spreads_df
    # spreads_test  = bt.test_spreads_df
    # returns_test   = bt.test_returns_df
    # returns_train   = bt.train_returns_df
    comp_weights=bt.comp_weights

    ## 2. Apply MRP optimizer to generate weights a mean-reverting portfolio 
    optimal_weights=MRP_optimizer_scipy(spreads_data=spreads_train)
    weights_by_levels=pd.concat([comp_weights.reset_index(level=1),optimal_weights],axis=1) # different levels of weights, combined by pair index

    weights_by_levels['flat_w']=weights_by_levels['component_w']*weights_by_levels['opt_w']
    weights_single_stocks=weights_by_levels.groupby('stock')['flat_w'].sum()

    
    ##  3. Run backtest using the optimized weights
    bt_MRP = backtest_MRP(weights_single_stocks, window_train, window_test)
    table_train, table_test = bt_MRP.run()

    return (table_train, table_test)

if __name__ == "__main__":

    table_train_list = []
    table_test_list = []

    num_month_train = 8
    num_month_test = 2

    dl = Data_loader(num_month_train=num_month_train, num_month_test=num_month_test)
    window_generator = dl.window_generator()
    windows = list(window_generator)

    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")

    start_time = time.time()

    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.starmap(pipeline_MRP, windows))

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Unpack the results into separate lists
    for result in results:
        table_train, table_test = result  # Unpack the tuple
        table_train_list.append(table_train)
        table_test_list.append(table_test)

    def save_files(table_train_list, table_test_list):
        output_dir = "output_MRP"
        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/train_tables.pkl', 'wb') as f:
            pickle.dump(table_train_list, f)
        
        with open(f'{output_dir}/test_tables.pkl', 'wb') as f:
            pickle.dump(table_test_list, f)
        
        print("Files saved using pickle")

    save_files(table_train_list, table_test_list)




