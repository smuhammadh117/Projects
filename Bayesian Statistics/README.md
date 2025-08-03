# UFC Fight Outcome Prediction using Bayesian Logistic Regression

This project applies **Bayesian logistic regression** to predict MMA (UFC) fight outcomes based on fighter statistics such as striking defense, knockout power, and ground impact. The analysis is implemented in **PyMC** using a custom Jupyter notebook.

## Project Structure

```
.
├── data/                   # Fighter stats and feature engineering (if any)
├── notebook.ipynb          # Main Bayesian modeling and evaluation notebook
├── report.pdf              # Full report explaining methodology and results
├── requirements.txt        # Required Python packages
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/smuhammadh117/Projects.git
cd Projects/Bayesian\ Statistics 
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebook (e.g., `notebook.ipynb`) in your browser.

## Methodology

- Logistic regression modeled using PyMC with binomial likelihood
- Fighter features include:
  - **SA**: Strikes Absorbed
  - **SD**: Striking Defense
  - **KO**: Knockout Power
  - **SS**: Standing Strikes
  - **GI**: Ground Impact (takedowns × submissions)

### Model Fit

- Four MCMC chains with 10,000 samples each, 1,000 burn-in
- Posterior inference using `arviz`
- Model sensitivity/specificity calculated
- Compared with a frequentist GLM approach using `statsmodels`

## Report

📄 Read the full project report [here](https://github.com/smuhammadh117/Projects/blob/main/Bayesian Statistics/UFCModelBayesianStatistics-2.pdf)

## References

- UFC Stats: [http://www.ufcstats.com/statistics/events/completed](http://www.ufcstats.com/statistics/events/completed)

