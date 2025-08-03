
import ManualStrategy as ml
import StrategyLearner as sl

import datetime as dt

import matplotlib.pyplot as plt
from util import get_data, plot_data
import pandas as pd
from marketsimcode import compute_portvals, port_stats


def experiment1(commission = 9.95, impact = 0.005):

    comm = commission
    imp = impact

    learner = sl.StrategyLearner(verbose=False, impact=imp, commission=comm)
    symbol = "JPM"
    train_start_date = dt.datetime(2008, 1, 1)
    train_end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000
    test_start_date = dt.datetime(2010, 1, 1)
    test_end_date = dt.datetime(2011, 12, 31)

    daterange_training = pd.date_range(train_start_date, train_end_date)
    daterange_testing = pd.date_range(test_start_date, test_end_date)

    manual_learner = ml.ManualLearner(verbose=False, impact=imp, commission=comm)
    manual_learner.add_evidence(symbol= symbol, sd = train_start_date, ed =train_end_date, sv = starting_value)
    datai,trades,long,short,mis, norm_benchmark = manual_learner.add_evidence(symbol= symbol, sd = train_start_date, ed =train_end_date, sv = starting_value)
    datao,trades,long,short,mos, norm_benchmark = manual_learner.add_evidence(symbol = symbol, sd =test_start_date, ed =test_end_date, sv = starting_value)


    mis = compute_portvals(mis, start_val=starting_value, commission=comm, impact=imp)
    mis = mis/mis.iloc[0]
    mos = compute_portvals(mos, start_val=starting_value, commission=comm, impact=imp)
    mos = mos/mos.iloc[0]


    learner.add_evidence(symbol=symbol, sd=train_start_date, ed=train_end_date, sv=starting_value)
    sis = learner.testPolicy(symbol=symbol, sd=train_start_date, ed=train_end_date, sv=starting_value)
    sos = learner.testPolicy(symbol=symbol, sd=test_start_date, ed=test_end_date, sv=starting_value)

    orders_sis = pd.DataFrame(index=sis.index, columns=['Symbol', 'Order', 'Shares'])
    orders_sis['Symbol'] = symbol
    orders_sis['Shares'] = sis
    for i in range(len(orders_sis)):
        if orders_sis['Shares'].iloc[i].squeeze() >0:
            orders_sis['Order'].values[i] = 'BUY'
        if orders_sis['Shares'].iloc[i].squeeze() <0:
            orders_sis['Order'].values[i] = 'SELL'

    orders_sis['Shares'] = abs(sis)



    orders_sos = pd.DataFrame(index=sos.index, columns=['Symbol', 'Order', 'Shares'])
    orders_sos['Symbol'] = symbol
    orders_sos['Shares'] = sos
    for i in range(len(orders_sos)):
        if orders_sos['Shares'].iloc[i].squeeze() >0:
            orders_sos['Order'].values[i] = 'BUY'
        if orders_sos['Shares'].iloc[i].squeeze() <0:
            orders_sos['Order'].values[i] = 'SELL'

    orders_sos['Shares'] = abs(sos)

    sis = compute_portvals(orders_sis, start_val=starting_value, commission=comm, impact=imp)
    sis = sis/sis.iloc[0]
    sos = compute_portvals(orders_sos, start_val=starting_value, commission=comm, impact=imp)
    sos = sos/sos.iloc[0]


    symbols = [symbol]
    prices_all_training = get_data(symbols, daterange_training)  # automatically adds SPY
    prices_training = prices_all_training[symbols]  # only portfolio symbols

    benchmark = 1000*prices_training
    norm_benchmark = benchmark/benchmark.iloc[0]

    norm_benchmark_training = norm_benchmark.loc[norm_benchmark.index.isin(daterange_training)]

    prices_all_training = get_data(symbols, daterange_testing)  # automatically adds SPY
    prices_testing = prices_all_training[symbols]  # only portfolio symbols

    benchmark = 1000*prices_testing
    norm_benchmark = benchmark/benchmark.iloc[0]

    norm_benchmark_testing = norm_benchmark.loc[norm_benchmark.index.isin(daterange_testing)]

    # Plot in sample data
    fig = plt.figure(figsize=(10, 6))
    plt.plot(datai, color="red", label="Manual Learner")
    plt.plot(sis, color="green", label="Strategy Learner")
    plt.plot(norm_benchmark_training, color="purple", label="Benchmark")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Portfolio Valuation", fontsize=12)
    plt.title("Experiment 1 in-sample, Commission = "+str(comm)+" Impact = "+str(imp), fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.savefig("Experiment1_in-sample")
    plt.clf()
    # plt.show()

    fig = plt.figure(figsize=(10, 6))
    plt.plot(datao, color="red", label="Manual Learner")
    plt.plot(sos, color="green", label="Strategy Learner")
    plt.plot(norm_benchmark_testing, color="purple", label="Benchmark")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Portfolio Valuation", fontsize=12)
    plt.title("Experiment 1 out-of-sample, Commission = "+str(comm)+" Impact = "+str(imp), fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.savefig("Experiment1_out-of-sample")
    plt.clf()
    # plt.show()

def author():
    return "shusainie3"

if __name__ == "__main__":
    experiment1()