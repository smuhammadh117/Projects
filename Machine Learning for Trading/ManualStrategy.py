
import indicators as ind
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from util import get_data, plot_data
import pandas as pd
from marketsimcode import compute_portvals, port_stats

class ManualLearner(object):

    def __init__(self, verbose=False, impact=0.0005, commission=9.95):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.bins = 5


    def add_evidence(self,symbol = "JPM", sd = dt.datetime(2008, 1, 1), ed = dt.datetime(2011, 12, 31), sv=10000):
        # Disregard why I called three separate times, it was just a mess up that got too much to fix
        symp = symbol
        symbols = [symbol]
        syms = [symbol]
        start_date = sd
        date_range = pd.date_range(sd,ed)
        prices_all = get_data(syms, date_range)  # automatically adds SPY
        prices = prices_all[syms]
        norm_prices = prices/prices.iloc[0]
        wind = 20

        # Bollinger Bands
        bb = ind.BollingerBands(norm_prices,window=wind)

        # SMA
        sma = ind.RollingMean(norm_prices, window = wind)

        # Momentum
        momentum = ind.Moment(norm_prices, window = wind)

        # Stochastic Oscillator
        k_smooth = ind.StochasticOscillator(norm_prices, N = 14, d_smooth = 3)

        # MACD
        macd = ind.MACD(norm_prices)

        dates = pd.date_range(sd, ed)
        prices_all = get_data(symbols, dates)  # automatically adds SPY
        prices = prices_all[symbols]  # only portfolio symbols
        norm_prices = norm_prices.loc[norm_prices.index.isin(dates)]
        daily_rets = np.zeros(prices.shape[0])
        daily_rets[1:] = (prices[1:].squeeze()/prices[:-1].squeeze().values)-1
        daily_rets = np.nan_to_num(daily_rets)
        tradelist = pd.DataFrame(index=norm_prices.index, columns=['Symbol', 'Order', 'Shares'])
        tradelist['Symbol'] = symp
        tradelist['Order'] = 'NOTHING'
        tradelist['Shares'] = 0

        sma = sma.loc[sma.index.isin(dates)]
        bb = bb.loc[bb.index.isin(dates)]
        momentum = momentum.loc[momentum.index.isin(dates)]
        k_smooth = k_smooth.loc[k_smooth.index.isin(dates)]
        macd = macd.loc[macd.index.isin(dates)]


        # Very fine-tuned discretization after several iterations, works well

        bins = self.bins


        bin1 = np.arange(0, bins, 1)
        try:
            discrete_k = pd.qcut(k_smooth.iloc[:, 0], q=5, labels=bin1)
        except ValueError as e:
            k_smooth.iloc[:, 0] = k_smooth.iloc[:, 0] + np.random.uniform(-1e-9, 1e-9, size=len(k_smooth.iloc[:, 0]))
            discrete_k = pd.qcut(k_smooth.iloc[:, 0], q=5, labels=bin1)

        discrete_k = discrete_k.fillna(0).astype(int)

        bin2 = np.arange(0, bins, 1)
        try:
            discrete_momentum = pd.qcut(momentum.iloc[:, 0], q=5, labels=bin2)
        except ValueError as e:
            momentum.iloc[:, 0] = momentum.iloc[:, 0] + np.random.uniform(-1e-9, 1e-9, size=len(momentum.iloc[:, 0]))
            discrete_momentum = pd.qcut(momentum.iloc[:, 0], q=5, labels=bin2)

        discrete_momentum = discrete_momentum.fillna(0).astype(int)

        bin3 = np.arange(0, bins, 1)
        bin3 = np.unique(bin3)
        discrete_sma = pd.qcut(sma.iloc[:, 0], q=5, labels=bin3)
        discrete_sma = discrete_sma.fillna(0).astype(int)

        bin4 = np.arange(0, bins, 1)
        bin4 = np.unique(bin4)
        discrete_macd = pd.qcut(macd.iloc[:, 0], q=5, labels=bin4)
        discrete_macd = discrete_macd.fillna(0).astype(int)

        bin5 = np.arange(0, bins, 1)
        bin5 = np.unique(bin5)
        discrete_bb = pd.qcut(bb.iloc[:, 0], q=5, labels=bin5)
        discrete_bb = discrete_bb.fillna(0).astype(int)


        indicators_df = pd.DataFrame({
            "BB": discrete_bb,
            "Momentum": discrete_momentum,
            "MACD": discrete_macd,
            "SMA" : discrete_sma,
            "Ksmooth": discrete_k
        })

        discretized_ind = indicators_df


        net_holdings = 0 # Starting value of shares
        net_holdings_tracker = []
        long = []
        short = []

        for i in range(1,len(daily_rets)):

            curr = discretized_ind.iloc[i]
            prev = discretized_ind.iloc[i - 1]

            # Sell conditions
            sell_condition = (
                    (net_holdings > -1000) and
                    (curr['MACD'] < 1 or curr['Momentum'] < 1) and
                    (prev['BB'] > prev['SMA'] and curr['BB'] < curr['SMA'] or prev['BB'] == 4) or
                    (curr['Ksmooth'] >3)
            )

            # Buy conditions
            buy_condition = (
                (net_holdings < 1000) and
                (curr['MACD'] > 3 or curr['Momentum'] > 3) and
                (prev['BB'] < prev['SMA'] and curr['BB'] > curr['SMA'] or prev['BB'] == 0) or
                (curr['Ksmooth'] < 1)
            )

            # Rules used for selling
            if sell_condition:

                if (net_holdings == 0):
                    stocks = -1000
                    net_holdings = net_holdings + stocks
                    net_holdings_tracker.append(net_holdings)
                    tradelist.loc[tradelist.index[i+1], 'Order'] = 'SELL'
                    tradelist.loc[tradelist.index[i+1], 'Shares'] = stocks
                    short.append(tradelist.index[i+1])

                if (net_holdings == 1000):
                    stocks = -2000
                    net_holdings = net_holdings + stocks
                    net_holdings_tracker.append(net_holdings)
                    tradelist.loc[tradelist.index[i+1], 'Order'] = 'SELL'
                    tradelist.loc[tradelist.index[i+1], 'Shares'] = stocks
                    short.append(tradelist.index[i+1])

            # Rules used for buying
            if buy_condition:

                if (net_holdings == 0):
                    stocks = 1000
                    net_holdings = net_holdings + stocks
                    net_holdings_tracker.append(net_holdings)
                    tradelist.loc[tradelist.index[i+1], 'Order'] = 'BUY'
                    tradelist.loc[tradelist.index[i+1], 'Shares'] = stocks
                    long.append(tradelist.index[i+1])

                if (net_holdings == -1000):
                    stocks = 2000
                    net_holdings = net_holdings + stocks
                    net_holdings_tracker.append(net_holdings)
                    tradelist.loc[tradelist.index[i+1], 'Order'] = 'BUY'
                    tradelist.loc[tradelist.index[i+1], 'Shares'] = stocks
                    long.append(tradelist.index[i+1])

        df_trades = tradelist[["Shares"]]
        orders = pd.DataFrame(index=df_trades.index.values, columns=['Symbol', 'Order', 'Shares'])


        orders['Symbol'] = symp

        orders['Order'] = "NOTHING"
        for i in range(len(df_trades)):
            if df_trades.iloc[i,0] > 0:
                orders['Order'].iloc[i] = "BUY"
            if df_trades.iloc[i,0] < 0:
                orders['Order'].iloc[i] = "SELL"

        trades = tradelist['Shares']
        trades.fillna(0, inplace=True)
        trades = trades.to_frame()

        orders['Shares'] = abs(df_trades)

        ms_port = compute_portvals(orders,start_val=sv, commission=self.commission, impact=self.impact)
        norm_ms_port = ms_port/ms_port.iloc[0]

        data = norm_ms_port
        # Compute the benchmark
        date_range = pd.date_range(sd,ed)
        prices_all = get_data(syms, date_range)  # automatically adds SPY
        prices = prices_all[syms]
        benchmark = 1000*prices
        norm_benchmark = benchmark/benchmark.iloc[0]
        norm_benchmark = norm_benchmark.loc[norm_benchmark.index.isin(date_range)]


        return data,trades,long,short,orders, norm_benchmark

    def testPolicy(self,symbol, sd,ed,sv):
        # Plot for training data
        symbol = symbol
        trainsd = dt.datetime(2008, 1, 1)
        trained = dt.datetime(2009,12,31)
        train_data, _, long_train, short_train, train_orders, norm_benchmark = self.add_evidence(symbol, trainsd, trained) # For table


        # Plot the Manual strategy vs benchmark
        fig = plt.figure(figsize=(10, 6))
        plt.plot(norm_benchmark, color="purple", label="Benchmark")
        plt.plot(train_data, color="red", label="Manual Strategy")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Normalized Portfolio Valuation", fontsize=12)
        plt.vlines(long_train, ymin=0, ymax=train_data.max(), color="blue", label="Long Position", linestyle = ":",  linewidth=1)
        plt.vlines(short_train, ymin=0, ymax=train_data.max(), color="black", label="Short Position", linestyle = ":",  linewidth=1)
        plt.title("Manual Strategy for in-sample data vs. Benchmark Portfolio Values", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.savefig("ManualStrategy_InSample")
        plt.clf()
        #
        # plt.show()


        # Plot for testing data
        symbol = "JPM"
        testsd = sd
        tested = ed
        test_data, trades, long_test, short_test, test_orders, norm_benchmark = self.add_evidence(symbol, testsd, tested) # For table

        # Plot the Manual strategy vs benchmark
        fig = plt.figure(figsize=(10, 6))
        plt.plot(norm_benchmark, color="purple", label="Benchmark")
        plt.plot(test_data, color="red", label="Manual Strategy")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Normalized Portfolio Valuation", fontsize=12)
        plt.vlines(long_test, ymin=0, ymax=test_data.max(), color="blue", label="Long Position", linestyle = ":",  linewidth=1)
        plt.vlines(short_test, ymin=0, ymax=test_data.max(), color="black", label="Short Position", linestyle = ":",  linewidth=1)
        plt.title("Manual Strategy for out-of-sample data vs. Benchmark Portfolio Values", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.savefig("ManualStrategy_OutofSample")
        plt.clf()


        cum_return_train, avg_daily_rets_train, std_daily_rets_train, sharpe_ratio_train = port_stats(train_data)
        cum_return_test, avg_daily_rets_test, std_daily_rets_test, sharpe_ratio_test = port_stats(test_data)

        performance_metrics = pd.DataFrame({
        "Metric": [
            "Cumulative Return",
            "Average Daily Return",
            "Standard Deviation of Daily Returns",
            "Sharpe Ratio"
        ],
        "In-sample": [
            cum_return_train,
            avg_daily_rets_train,
            std_daily_rets_train,
            sharpe_ratio_train
        ],
        "Out-of-sample": [
            cum_return_test,
            avg_daily_rets_test,
            std_daily_rets_test,
            sharpe_ratio_test
        ]
        })

        performance_metrics.to_csv('performance_metricsManualStrategy.txt', sep='\t', index=False)
        with open('performance_metricsManualStrategy.txt', 'w') as f:
            f.write(performance_metrics.to_string(index=False))


        return trades


    def author(self):
        return "shusainie3"


if __name__ == "__main__":
    print("League of legends")
    # learner = ManualLearner(verbose=True, impact=0.005, commission=9.95)
    # k,l = learner.testPolicy(symbol = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
