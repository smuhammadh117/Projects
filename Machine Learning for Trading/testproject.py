
import datetime as dt
import ManualStrategy as ml
import StrategyLearner as sl
import experiment1 as exp1
import experiment2 as exp2

if __name__ == "__main__":

    ms = ml.ManualLearner(impact = 0.005, commission=9.95)
    df_trades = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)


    learner = sl.StrategyLearner(verbose = False, impact = 0.005, commission=9.95) # constructor
    learner.add_evidence(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase

    exp1.experiment1(commission=9.95,impact=0.005)

    exp2.experiment2()


def author():
    return "shusainie3"




