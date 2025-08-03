# Strategy Evaluation for Stock Trading (Project 8)

This project implements and evaluates both **manual rule-based** and **machine learning-based trading strategies** using a **Random Tree Bag Learner** to generate buy/sell signals. It simulates realistic trading scenarios using historical stock data and calculates key performance metrics like Sharpe Ratio and Rolling Returns.

Note: Initial framework code was provided by Georgia Tech OMSCS course (CS 7646). Although a lot of additions were made for the code to work, for copyright reasons only fully student-implemented modules and analysis are shared. Access can be provided to the private repository files if/ when needed


## Project Structure

```
├── StrategyLearner.py              # Main strategy learner with Bag Learner
├── RTLearner.py                    # Implements a random tree learner 
├── BagLearner.py                   # Implements a random tree learner multiple times to create a random tree forest
├── indicators.py                   # Technical indicators (MACD, BB, SMA, Momentum, etc.)
├── marketsimcode.py                # Simulates portfolio values
├── experiment1.py                  # To compare the manual rules to the strategy learner
├── experiment2.txt                 # To evaluate the effect of stock market impact on the strategy learner
├── testproject.py                  # To run both experiments 1 and 2
├── util.py                         # Utility functions for fetching and processing data
├── p8_strategyEval_report-2.pdf    # Project writeup/report
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/smuhammadh117/Projects.git
   cd Projects/Reinforcement\ Learning/project_8
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Run testproject.py

```bash
python testproject.py
```

This script will:
- Train the `StrategyLearner` at multiple impact levels
- Compare performance against a buy-and-hold benchmark
- Save a performance plot and results to `Experiment2_in-sample.png` and `Experiment2.txt`

## Output Metrics

- **Sharpe Ratio**: Risk-adjusted return of the strategy
- **100-day Rolling Return**: Measures smoothness and profitability over time
- **Benchmark Comparison**: Evaluates against passive investing

## Indicators Used

- **SMA** – Simple Moving Average
- **BB** – Bollinger Bands
- **Momentum**
- **Stochastic Oscillator**
- **MACD** – Moving Average Convergence Divergence

Each indicator was discretized to reduce overfitting and enable meaningful decision boundaries for learning algorithms.

## Report

Refer to `p8_strategyEval_report-2.pdf` for the full explanation, methodology, and results visualizations.
