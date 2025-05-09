import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import List, Tuple, Dict


class PairTradingBacktester_TrainTest:
    """
    A class for backtesting pair trading strategies using training and testing datasets.
    - Uses training data to rank pairs by Sharpe ratio and selects the top-k performing pairs.
    - Re-runs these top-k pairs on the test data with the same parameters to assess out-of-sample performance.
    - Retains intermediate results (e.g., spread, returns, profit per day) for both datasets.

    After calling .run():
        - self.summary: DataFrame with ranks and Sharpe ratios for both train and test sets.
        - self.train_results: Dictionary mapping each pair to its backtest results on the training set.
        - self.test_results: Dictionary mapping each pair to its backtest results on the testing set.
    """

    # ------------------------------------------------------------------ #
    # ---------------------------  init  --------------------------------#
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        params: List[Tuple[float, float, float]],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        trading_days_per_year: int = 252
    ):
        """
        Initializes the pair trading backtester.

        Parameters:
        - pairs: List of tuples, each containing two stock symbols forming a trading pair (e.g., ("AAPL", "MSFT")).
        - params: List of tuples, each containing (beta, mu, sigma2) for a pair, where:
            - beta: Cointegration coefficient between the pair's stock prices.
            - mu: Mean of the residual in the cointegration model.
            - sigma2: Variance of the residual.
        - train_data: DataFrame with training period stock prices, indexed by date, columns as stock symbols.
        - test_data: DataFrame with testing period stock prices, indexed by date, columns as stock symbols.
        - trading_days_per_year: Number of trading days in a year for annualizing metrics (default: 252).

        Instance Variables:
        - self.train_results: Populated by .run() with backtest results for each pair on training data.
        - self.test_results: Populated by .run() with backtest results for each pair on testing data.
        - self.summary: Populated by .run() with a summary of ranks and Sharpe ratios.
        - self.train_spreads_df, self.test_spreads_df: DataFrames with spreads for top-k pairs.
        - self.train_returns_df, self.test_returns_df: DataFrames with returns for top-k pairs.
        - self.comp_weights: MultiIndex Series with component weights for top-k pairs' stocks.
        """
        assert len(pairs) == len(params), "`pairs` and `params` length mismatch"
        self.pairs = pairs
        self.params = params
        self.train_data = train_data
        self.test_data = test_data
        self.trading_days = trading_days_per_year

        # filled by .run()
        self.train_results: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.test_results: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.summary: pd.DataFrame | None = None
        # --------- extra artefacts for top-k pairs ---------------- #
        self.train_spreads_df: pd.DataFrame | None = None
        self.test_spreads_df: pd.DataFrame | None = None
        self.train_returns_df: pd.DataFrame | None = None
        self.test_returns_df: pd.DataFrame | None = None

        # Component weights for constructing the spread: spread = w1*stock1 + w2*stock2
        # where w1 = 1 / (1 - beta), w2 = -beta / (1 - beta)
        # Stored as a MultiIndex Series for later use in exposure calculations
        self.comp_weights = None

    # ------------------------------------------------------------------ #
    # --------------------   helper functions   ------------------------ #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _threshold_gaussian(sigma: float) -> float:
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

    # -- single-pair runner (identical logic, data passed explicitly) --- #
    def _backtest_single(
        self,
        pair: Tuple[str, str],
        param: Tuple[float, float, float],
        data: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Backtests a single trading pair using the provided parameters and dataset.

        Parameters:
        - pair: Tuple of two stock symbols (e.g., ("AAPL", "MSFT")).
        - param: Tuple of (beta, mu, sigma2) for the pair.
        - data: DataFrame with stock prices, indexed by date, columns as stock symbols.

        Returns:
        - Dictionary with backtest results: trade table, spread series, return series,
          profit per day, and Sharpe ratio.

        Process:
        1. Calculates residuals and spread using the cointegration model.
        2. Generates trade and unwind signals based on thresholds.
        3. Simulates trading to compute returns and portfolio value.
        4. Computes the annualized Sharpe ratio.
        """
        stock1, stock2 = pair
        beta, mu, sigma2 = param

        # ----------------  stage 0: core columns  ---------------- #
        # Residuals: Deviation from the cointegration relationship
        residuals = data[stock1] - beta * data[stock2] - mu
        # Spread: Portfolio value of the pair (stock1 - beta * stock2)
        spread = data[stock1] - beta * data[stock2]

        # Initialize DataFrame with core data
        tbl = pd.DataFrame(
            {
                "residual": residuals,
                "stock1_price": data[stock1],
                "stock2_price": data[stock2],
                "beta": beta,
                "mu": mu,
                "spread": spread,
            },
            index=data.index,
        )

        # ----------------  stage 1: signals  --------------------- #
        # Calculate trading threshold based on residual standard deviation
        thr = self._threshold_gaussian(np.sqrt(sigma2))
        # Trade signal: -1 (short) if residual >= threshold, 1 (long) if <= -threshold, else 0
        tbl["trade_signal"] = np.where(
            residuals >= thr, -1, np.where(residuals <= -thr, 1, 0)
        )
        # Unwind signal: 1 if residual sign changes (indicating mean reversion), else 0
        tbl["unwind_signal"] = (np.sign(residuals) != np.sign(residuals).shift()).astype(int)
        tbl.iloc[0, tbl.columns.get_loc("unwind_signal")] = 0  # No unwind on first day

        # ----------------  stage 2: P&L loop  -------------------- #
        pos, prev_price, port_val = 0, np.nan, 1.0  # Initial position, previous spread, portfolio value
        pos_list, pv_list, ret_list = [], [], []  # Lists for position, portfolio value, returns

        # Iterate through each day to simulate trading
        for _, row in tbl.iterrows():
            if pos == 0:  # No position
                if row["trade_signal"] != 0:
                    pos = row["trade_signal"]  # Enter long (1) or short (-1) position
                    prev_price = row["spread"]  # Record entry price
                ret_list.append(0.0)  # No return when flat
            elif pos == 1:  # Long position
                cur = row["spread"]
                ret = cur / prev_price  # Return = current spread / entry spread
                port_val *= ret  # Update portfolio value
                prev_price = cur  # Update previous price
                ret_list.append(ret - 1.0)  # Daily return
                if row["unwind_signal"]:
                    pos = 0  # Exit position on unwind signal
            elif pos == -1:  # Short position
                cur = row["spread"]
                ret = 2.0 - cur / prev_price  # Short return = 2 - (current / entry)
                port_val *= ret  # Update portfolio value
                prev_price = cur  # Update previous price
                ret_list.append(ret - 1.0)  # Daily return
                if row["unwind_signal"]:
                    pos = 0  # Exit position on unwind signal
            else:
                raise ValueError("Invalid position state")

            pos_list.append(pos)
            pv_list.append(port_val)

        # Add simulation results to the table
        tbl["position"] = pos_list
        tbl["portfolio_val"] = pv_list
        tbl["ret"] = ret_list
        # Profit per day: Return adjusted by position direction
        tbl["profit_per_day"] = tbl["ret"] * tbl["position"]

        # ----------------  stage 3: Sharpe  ---------------------- #
        full_ret = np.asarray(ret_list, float)
        mask = ~np.isnan(full_ret)
        if mask.sum() > 1 and np.std(full_ret[mask]) > 0:
            # Annualized Sharpe ratio: mean return / std dev * sqrt(trading days)
            sharpe = (
                np.mean(full_ret[mask])
                / np.std(full_ret[mask])
                * np.sqrt(self.trading_days)
            )
        else:
            sharpe = 0.0  # Default to 0 if insufficient data

        # Return all backtest artifacts
        return {
            "trade_table": tbl,
            "spread_series": tbl["spread"].copy(),
            "ret_series": tbl["ret"].copy(),
            "profit_per_day": tbl["profit_per_day"].copy(),
            "sharpe": sharpe,
        }

    # ------------------------------------------------------------------ #
    # ------------------------   public API   -------------------------- #
    # ------------------------------------------------------------------ #
    def run(self, top_k: int = 50) -> pd.DataFrame:
        """
        Executes the full backtesting process across train and test datasets.

        Parameters:
        - top_k: Number of top pairs to select based on training Sharpe ratio (default: 50).

        Returns:
        - DataFrame with columns: orig_rank, pair, train_sharpe, train_rank, test_sharpe, test_rank.

        Steps:
        1. Backtests all pairs on training data and collects Sharpe ratios.
        2. Ranks pairs by training Sharpe and selects the top-k.
        3. Backtests the top-k pairs on test data with the same parameters.
        4. Assembles panel data (spreads, returns) and component weights for top-k pairs.
        """
        # -------- step 1: train run & collect Sharpe --------------- #
        records = []
        for idx in tqdm(range(len(self.pairs)), desc="Train back-test"):
            pair = self.pairs[idx]
            res_train = self._backtest_single(pair, self.params[idx], self.train_data)
            self.train_results[pair] = res_train
            records.append(
                {
                    "pair": pair,
                    "orig_rank": idx + 1,  # 1-based index in original pairs list
                    "train_sharpe": res_train["sharpe"],
                }
            )

        # Sort pairs by training Sharpe ratio
        df = pd.DataFrame(records).sort_values("train_sharpe", ascending=False)
        df["train_rank"] = np.arange(1, len(df) + 1)

        # -------- step 2: keep top-k -------------------------------- #
        top_df = df.head(top_k).copy()
        selected = set(map(tuple, top_df["pair"]))

        # -------- step 3: test run for selected pairs --------------- #
        for pair in tqdm(selected, desc="Test back-test"):
            idx = self.pairs.index(pair)  # Reuse original parameters
            self.test_results[pair] = self._backtest_single(
                pair, self.params[idx], self.test_data
            )
            top_df.loc[top_df["pair"] == pair, "test_sharpe"] = self.test_results[pair]["sharpe"]

        # Assign test ranks based on test Sharpe ratios
        top_df['test_rank'] = top_df['test_sharpe'].rank(method='min', ascending=False)

        # Store the summary table
        self.summary = top_df.reset_index(drop=True)

        # ---------------------------------------------------------- #
        #   assemble panel data & component weights for top-k pairs  #
        # ---------------------------------------------------------- #
        top_pairs = list(self.summary["pair"])  # List of top-k pairs

        def _panel(field: str, which: str) -> pd.DataFrame:
            """
            Creates a panel DataFrame for a given field (e.g., spread_series) from train or test results.

            Parameters:
            - field: The result field to extract (e.g., "spread_series", "ret_series").
            - which: Dataset to use ("train" or "test").

            Returns:
            - DataFrame with columns as pairs and rows as dates.
            """
            res_dict = self.train_results if which == "train" else self.test_results
            cols = []
            for p in top_pairs:
                s = res_dict[p][field].rename(p)  # Column name is the pair tuple
                cols.append(s)
            return pd.concat(cols, axis=1)

        # Populate panel DataFrames for spreads and returns
        self.train_spreads_df = _panel("spread_series", "train")
        self.test_spreads_df = _panel("spread_series", "test")
        self.train_returns_df = _panel("ret_series", "train")
        self.test_returns_df = _panel("ret_series", "test")

        # Compute component weights for top-k pairs
        self.comp_weights = self.get_component_weights()

        return self.summary

    # convenience getters ---------------------------------------------- #
    def get_pair_table(
        self, pair: Tuple[str, str], dataset: str = "train"
    ) -> pd.DataFrame:
        """
        Retrieves the detailed trade table for a specific pair.

        Parameters:
        - pair: Tuple of two stock symbols (e.g., ("AAPL", "MSFT")).
        - dataset: Dataset to query ("train" or "test", default: "train").

        Returns:
        - DataFrame with trade details (e.g., residual, spread, position, returns).
        """
        if dataset == "train":
            return self.train_results[pair]["trade_table"]
        elif dataset == "test":
            return self.test_results[pair]["trade_table"]
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    def get_pair_series(
        self, pair: Tuple[str, str], field: str, dataset: str = "train"
    ) -> pd.Series:
        """
        Retrieves a specific time series for a pair from the backtest results.

        Parameters:
        - pair: Tuple of two stock symbols (e.g., ("AAPL", "MSFT")).
        - field: Series to retrieve ("spread_series", "ret_series", or "profit_per_day").
        - dataset: Dataset to query ("train" or "test", default: "train").

        Returns:
        - Pandas Series with the requested data, indexed by date.
        """
        results = self.train_results if dataset == "train" else self.test_results
        return results[pair][field]

    def get_component_weights(self) -> pd.Series:
        """
        Computes component weights for each stock in the top-k pairs.

        Returns:
        - MultiIndex Series with index (pair, stock) and values as weights.
          - Weights define the spread: spread = w1*stock1 + w2*stock2.
          - w1 = 1 / (1 - beta), w2 = -beta / (1 - beta).
          - Useful for calculating total stock exposures with pair-level weights:
            total_stock_w = comp_w.mul(pair_w, level=0).groupby(level=1).sum().
        """
        # Map each pair to its beta parameter
        beta_map = dict(zip(self.pairs, (b for b, *_ in self.params)))

        tuples, vals = [], []
        for pair in self.summary["pair"]:
            s1, s2 = pair
            beta = beta_map[pair]
            # w1 = 1.0 / (1.0 - beta)  # Weight for stock1
            # w2 = -beta / (1.0 - beta)  # Weight for stock2
            w1 = 1
            w2 = -beta

            # Add weights for both stocks in the pair
            tuples.extend([(pair, s1), (pair, s2)])
            vals.extend([w1, w2])

        # Create MultiIndex Series
        idx = pd.MultiIndex.from_tuples(tuples, names=["pair", "stock"])
        return pd.Series(vals, index=idx, name="component_w")