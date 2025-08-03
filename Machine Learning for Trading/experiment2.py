
import StrategyLearner as sl
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from util import get_data, plot_data
import pandas as pd
from marketsimcode import compute_portvals, port_stats



def experiment2():

    symbol = "JPM"
    train_start_date = dt.datetime(2008, 1, 1)
    train_end_date = dt.datetime(2010, 5, 31)
    starting_value = 100000


    comm = 0
    impacts = [0.0, 0.005, 0.01, 0.02]
    daterange_training = pd.date_range(train_start_date, train_end_date)

    symbols = [symbol]
    prices_all_training = get_data(symbols, daterange_training)  # automatically adds SPY
    prices_training = prices_all_training[symbols]  # only portfolio symbols

    benchmark = 1000*prices_training
    norm_benchmark = benchmark/benchmark.iloc[0]

    norm_benchmark = norm_benchmark.loc[norm_benchmark.index.isin(daterange_training)]

    final_portval = []
    portfolio_values = {}
    sr_values = []

    for impact in impacts:
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=comm)
        learner.add_evidence(symbol=symbol, sd=train_start_date, ed=train_end_date, sv=starting_value)
        trades = learner.testPolicy(symbol=symbol, sd=train_start_date, ed=train_end_date, sv=starting_value)

        orders = pd.DataFrame(index=trades.index, columns=['Symbol', 'Order', 'Shares'])
        orders['Symbol'] = symbol
        orders['Shares'] = trades
        for i in range(len(orders)):
            if orders['Shares'].iloc[i].squeeze() >0:
                orders['Order'].values[i] = 'BUY'
            if orders['Shares'].iloc[i].squeeze() <0:
                orders['Order'].values[i] = 'SELL'

        orders['Shares'] = abs(trades)

        portvals = compute_portvals(orders, start_val=starting_value, commission=comm, impact=impact)
        portvals = portvals/portvals.iloc[0]
        portfolio_values[impact] = portvals
        final_portval.append((portvals.iloc[-1].squeeze()/portvals.iloc[-101].squeeze()) - 1)

        prices = portvals
        daily_rets = np.zeros(prices.shape[0])
        daily_rets[1:] = (prices[1:].squeeze()/prices[:-1].squeeze().values)-1
        daily_rets = np.nan_to_num(daily_rets)
        adr = np.mean(daily_rets)
        sddr = np.std(daily_rets)
        sr = np.sqrt(252)*adr/sddr
        sr_values.append(sr)

    plt.figure(figsize=(12, 6))

    handles = []
    labels = []
    for imp, portvals_norm in portfolio_values.items():
        line, = plt.plot(portvals_norm, label=f"Impact = {imp:.3f}")
        handles.append(line)
        labels.append(f"Impact = {imp:.3f}")

    line, = plt.plot(norm_benchmark, color="purple", label="Benchmark")
    handles.append(line)
    labels.append("Benchmark")

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Portfolio Valuation", fontsize=12)
    plt.title("Experiment 2 in-sample", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend(handles, labels, loc="lower left")

    plt.savefig("Experiment2_in-sample")
    plt.clf()



    results = pd.DataFrame({
        "Metric": [
        "Impact 0",
        "Impact 0.005",
        "Impact 0.01",
        "Impact 0.02"
    ],
    "Sharpe ratio": [
        sr_values[0],
        sr_values[1],
        sr_values[2],
        sr_values[3]
    ],
    "100 day rolling return": [
        final_portval[0],
        final_portval[1],
        final_portval[2],
        final_portval[3]
    ]
    })


    results.to_csv('Experiment2.txt', sep='\t', index=True)
    with open('Experiment2.txt', 'w') as f:

        f.write(results.to_string(index=False))


def author():
    return "shusainie3"

if __name__ == "__main__":
    experiment2()