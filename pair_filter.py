import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import math

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression



# normalized price distance filter (initial filter)
class NPD_filter():
    def __init__(self, df_stock_price):

        initial_price = df_stock_price.iloc[0]
        normalized_price = df_stock_price.div(initial_price)

        self.price_distance_list = self.__calculate_squared_distances(normalized_price)

    
    def select_pairs(self, percentile=5, verbose=False):
        """
        get top [percentile]% of paris with smallest distance
        """

        percentile  = percentile/100
        idx = math.floor(percentile*len(self.price_distance_list))
        selected_pairs = self.price_distance_list[0:idx]

        pairs = [element[0] for element in selected_pairs]
        distances = [element[1] for element in selected_pairs]

        if verbose:
            print(f"{len(selected_pairs)} pairs selected")
            print(f"min distance {selected_pairs[0][1]}")
            print(f"max distance {selected_pairs[-1][1]}")
        
        return pairs, distances



    def visualize_price_distance(self, visualize_percentile=10):


        distances = np.array([item[1] for item in self.price_distance_list])
        print("Basic Statistics for distances:")
        print(f"Mean: {np.mean(distances)}")
        print(f"Median: {np.median(distances)}")
        print(f"Standard Deviation: {np.std(distances)}")

        percentiles = [0, 25, 50, 75, 100]
        percentile_values = np.percentile(distances, percentiles)

        print("\nPercentiles:")
        for p, val in zip(percentiles, percentile_values):
            print(f"{p}th percentile: {val}")

        # Min and Max
        print(f"\nMinimum: {np.min(distances)}")
        print(f"Maximum: {np.max(distances)}")

        # visualize first 10 percentiles
        percentile = visualize_percentile

        percentile_value = np.percentile(distances, percentile)
        vals = distances[distances <= percentile_value]

        plt.hist(vals, bins=30)
        plt.xlabel('normalized price distances')
        plt.ylabel('count')
        plt.title(f'NPD distribution,  smallest {percentile} percentile')
        plt.show()



    def __calculate_squared_distances(self, df):
        """
        返回一个list,每一个list为一个tuple:
        ((stock1_ticker, stock2_ticker), distance)

        list按照distance由小到大排序
        """

        df_transposed = df.T
        
        # Calculate pairwise distances and square them
        pairwise_distances = pdist(df_transposed, metric='euclidean')
        squared_distances = np.square(pairwise_distances)
        # Convert the condensed distance matrix to a square matrix
        squared_distance_matrix = squareform(squared_distances)
        
        # Create a DataFrame with stock names as indices and columns
        stocks = df_transposed.index
        distance_df = pd.DataFrame(
            squared_distance_matrix,
            index=stocks,
            columns=stocks
        )

        # Convert matrix to dictionary
        distance_dict = {}
        for i, stock1 in enumerate(stocks):
            for j, stock2 in enumerate(stocks):
                if i < j:
                    pair = (stock1, stock2)
                    distance_dict[pair] = distance_df.iloc[i, j]

        price_distance_list = sorted(distance_dict.items(), key=lambda x: x[1])

        return price_distance_list
    
    





# OLS filter (second layer after NPD)
class OLS_filter():

    def __init__(self, df_stock_price):

        self.stock_data = df_stock_price

    
    def select_pairs(self, pairs, verbose=False, confidence_level=0.05):
        test_result, test_params = self.test_pairs(pairs, confidence_level=confidence_level)

        if verbose:
            total_num = len(test_result)
            passed_num = np.sum(test_result)
            print(f"{passed_num} out of {total_num} pairs passed the test")
            print(f"pass rate: {passed_num/total_num:.2f}")

        selected_pairs = []
        selected_params = []
        for idx in range(len(pairs)):
            if test_result[idx]:
                selected_pairs.append(pairs[idx])
                selected_params.append(test_params[idx])

        return selected_pairs, selected_params


    def test_pairs(self, pairs, confidence_level=0.05):
        test_result = []
        test_params = []

        for pair in pairs:
            stock1 = pair[0]
            stock2 = pair[1]
            is_stationary, params = self.test_one_pair(stock1, stock2, verbose=False, confidence_level=confidence_level)
            test_result.append(is_stationary)
            test_params.append(params)
        

        return test_result, test_params


    def test_one_pair(self, stock1, stock2, verbose=False, confidence_level=0.05):

        y = np.array(self.stock_data[stock1])
        X = np.array(self.stock_data[stock2]).reshape(-1, 1)

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        residual_variance = np.var(residuals, ddof=1)

        is_stationary = self.__test_residuals_ADF(residuals, confidence_level=confidence_level)

        if verbose:
            print(f"Residuals are stationary: {is_stationary}")
            print("Coefficient:", model.coef_[0])
            print("Intercept:", model.intercept_)
            print("Residual variance:", residual_variance)

        # 返回OLS的参数，避免重复计算(beta, mu, sigma^2)
        params = (model.coef_[0], model.intercept_, residual_variance)
        return is_stationary, params


    def __test_residuals_ADF(self, residuals, confidence_level=0.05):
        """
        Augmented Dickey-Fuller test: testing if residuals are stationary
        """

        # 很奇怪,某些股票,如MALTZ和CXH的价格一直是常数
        # 这时候residuals也是个常数, adfuller test会报错
        # 所以我们设置判断条件,直接输出p-val1
        if max(residuals) == min(residuals):
            p_val = 1
        else:
            result = adfuller(residuals)
            p_val = result[1]
        
        # ADF test的一些output
        # print('ADF Statistic:', result[0])
        # print('p-value:', result[1])
        # print('Critical Values:')
        # for key, value in result[4].items():
        #     print(f'   {key}: {value}')
        
    
        # 也可以加入其他判断逻辑
        # 比如需要residuals的variance大于一定的threshold
        # 甚至是限制两个股票属于同一个行业

        # 当p-val小于confidence level时,说明residuals是stationary
        res = True if p_val < confidence_level else False

        return res
    