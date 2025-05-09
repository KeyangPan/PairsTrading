
import pandas as pd
import numpy as np
import math


class Data_loader():
    def __init__(self, 
                 train_ratio = 2/3,
                 window_length = math.floor(253*1.5),
                 nasdaq_path = 'data/nasdaq_19900101_20031231.csv', 
                 nyse_path = 'data/nyse_19900101_20031231.csv'):

        self.window_length = window_length
        self.train_ratio = train_ratio

        self.nasdaq_path = nasdaq_path
        self.nyse_path = nyse_path

        # 读取两个交易所的股票数据
        self.nasdaq_data = self.__read_data(nasdaq_path)
        self.nyse_data = self.__read_data(nyse_path)

        assert self.__check_duplicate_stocks(self.nasdaq_data) is True, "duplicate columns in nasdaq data"
        assert self.__check_duplicate_stocks(self.nyse_data) is True, "duplicate columns in nyse data"


        self.combined_data = self.__merge_exchange()
        assert self.__check_duplicate_stocks(self.combined_data) is True, "duplicate columns in combined data"


    def get_window(self, start_idx, window_length=None, verbose=False):
        """
        从start index开始, 一年半的window,
        删除含有nan值的股票,只保留有完整数据的股票列
        """

        if window_length is None:
            window_length = self.window_length

        length_check = (start_idx + window_length <= self.combined_data.shape[0])
        assert length_check, "start_idx out of bound"


        res = self.combined_data.iloc[start_idx:start_idx+window_length]

        original_width = res.shape[1]
        res = res.dropna(axis=1) #去除有nan的股票列

        new_width = res.shape[1]
        cols_dropped = original_width - new_width

        if verbose:
            window = f"window with start: {res.index[0].strftime('%Y-%m-%d')}, end: {res.index[-1].strftime('%Y-%m-%d')}"
            dropped = f"{cols_dropped} columns dropped due to NaN value"
            print(window)
            print(dropped)

        res = res.sort_index()

        start_index = math.floor(len(res) * self.train_ratio)

        window_train = res.iloc[0:start_index]
        window_test = res.iloc[start_index:]

        return window_train, window_test
    
    def window_loader(self):
        # 定义一个iterator,返回一个iterator
        # 每半年返回一个窗口用于train和test
        # TODO
        return 



    # 合并两个交易所的股票数据
    def __merge_exchange(self):

        assert (self.nasdaq_data.index == self.nyse_data.index).all()
        stock_data = pd.concat([self.nasdaq_data, self.nyse_data], axis=1)

        # 某些股票在nasdaq和nyse都有交易
        # 这种情况下,我们只保留nasdaq的数据
        mask = ~stock_data.columns.duplicated(keep='first')
        stock_data = stock_data.loc[:, mask]
        return stock_data




    def __read_data(self, filepath):
        data = pd.read_csv(filepath)

        # adjust price for dividend/split
        data['price'] = data['PRC'].abs() * data['CFACPR']

        data['date'] = pd.to_datetime(data['date'])

        # pivot using PERMNO as columns
        data_pivot = data.pivot_table(
            index='date',
            columns='PERMNO',
            values='price',
            # aggfunc='first'  # Use first value if there are duplicates
        )

        # Get the most recent ticker for each PERMNO (since tickers can change over time)
        permno_to_ticker = data.sort_values('date').drop_duplicates('PERMNO', keep='last')[['PERMNO', 'TICKER']]
        permno_to_ticker_dict = dict(zip(permno_to_ticker['PERMNO'], permno_to_ticker['TICKER']))

        # Rename the columns using the ticker symbols
        data_pivot.columns = [permno_to_ticker_dict.get(permno, str(permno)) for permno in data_pivot.columns]

        # 说明：
        # PERMNO是唯一的,但是有些不同股票共用了同一个ticker名称,
        # 比如EYE对应了三个PERMNO
        # 为了避免冲突，直接drop所有重名问题的股票
        mask = ~ data_pivot.columns.duplicated(keep=False)
        cols_keep = data_pivot.columns[mask]
        data_pivot = data_pivot[cols_keep]

        return data_pivot

    # 以下代码可以检测columns是否unique
    def __check_duplicate_stocks(self, df, verbose=False):
        stocks_original = len(list(df.columns))
        unique = len(set(df.columns))

        if verbose:
            print(f"number of duplicate tickers: {stocks_original - unique}")
        
        no_duplicate = (stocks_original-unique == 0)
        return no_duplicate

    


    