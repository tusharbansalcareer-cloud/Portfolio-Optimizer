# Portfolio Optimization & Backtesting Dashboard

Streamlit dashboard for Indian-equity portfolio construction, strategy comparison, Monte Carlo risk simulation, and strict walk-forward backtesting.

The app is built for a practical investing workflow: choose stocks, optimize a portfolio, validate it against a benchmark, inspect risk and diversification, simulate possible future outcomes, then test whether the same decision rule would have worked historically without look-ahead bias.

## What The App Does

- Resolves common Indian stock and index names into Yahoo Finance tickers, such as `reliance` -> `RELIANCE.NS` and `nifty` -> `^NSEI`.
- Downloads live historical price data through `yfinance`.
- Cleans price data and converts it into daily simple returns.
- Builds long-only portfolios with multiple allocation strategies.
- Supports optional manual/custom allocation with percentage weights.
- Compares strategies using the same return, risk, drawdown, turnover, and benchmark metrics.
- Runs an alpha and information-ratio gate before showing Monte Carlo risk output for the selected portfolio.
- Shows portfolio allocation, risk contribution, annual returns, cumulative returns, worst drawdown periods, and correlation matrix.
- Runs Monte Carlo simulations for a selected time horizon and initial capital.
- Runs static optimized-weight backtests or walk-forward backtests that retrain at every rebalance using only trailing historical data.
- Applies transaction costs and slippage during backtests.
- Tracks optimizer fallback events and exposes optional debug diagnostics.

## Main Screens

### 1. Portfolio Optimization

This tab is the main current-allocation workflow.

You enter:

- Stock names or Yahoo tickers.
- Start and end dates for historical data.
- Benchmark name or ticker, defaulting to `^NSEI`.
- Optimization strategy.
- Risk-free rate.
- Maximum single-stock weight.
- Alpha and information-ratio thresholds.
- Monte Carlo simulation settings.
- Initial capital.
- Optional L2 regularization.
- Optional custom manual weights.

When you run the portfolio pipeline, the app:

1. Validates the inputs.
2. Resolves names into Yahoo Finance tickers.
3. Loads historical adjusted-close data.
4. Computes clean daily returns.
5. Runs every optimizer strategy.
6. Adds custom manual allocation when enabled.
7. Selects the requested strategy.
8. Compares the selected portfolio against the benchmark.
9. Runs the alpha gate.
10. Renders allocation, return, risk, drawdown, correlation, and Monte Carlo outputs.

The output includes:

- Alpha, beta, information ratio, and tracking error.
- Portfolio period return vs benchmark period return.
- Expected annual return, annual volatility, Sharpe ratio, and optimizer source.
- Portfolio weights table.
- Allocation pie chart.
- Worst drawdown-period chart.
- Annual return chart.
- Cumulative return vs benchmark chart.
- Correlation heatmap and matrix for all tickers.
- Risk contribution chart.

### 2. Strategy Comparison

This tab compares all available strategies on the same cleaned return history from the most recent portfolio run.

Strategies currently include:

- `Mean-Variance / Max Sharpe`
- `Minimum Variance`
- `Risk Parity`
- `Equal Weight`
- `Custom Manual Allocation`, when enabled

The comparison view shows:

- Summary table for returns, volatility, Sharpe, drawdown, turnover, and related metrics.
- Strategy cumulative-return chart.
- Strategy annual-return chart.
- Weights by strategy.
- Risk contribution by strategy.

This makes it easier to see whether a high-return strategy is taking concentration risk, volatility risk, or unstable allocations compared with simpler alternatives.

### 3. Backtesting

This tab can test the latest optimized allocation as a static hold, or test strategy rules historically using a walk-forward process.

When rebalance frequency is `None`, the app holds the supplied optimized or manual weights through the full selected backtest period and applies entry transaction cost once.

For Weekly, Monthly, and Quarterly frequencies, the app does not reuse today's optimized weights throughout history. Instead, for each rebalance date, it:

1. Looks backward using the selected trailing lookback window.
2. Optimizes using only data available before that rebalance date.
3. Holds the resulting weights until the next rebalance.
4. Applies turnover-based transaction cost and slippage.
5. Records realized out-of-sample performance.

Supported rebalance frequencies:

- None
- Weekly
- Monthly
- Quarterly

Supported lookback windows:

- 6 Months / 126 trading days
- 1 Year / 252 trading days
- 2 Years / 504 trading days
- 3 Years / 756 trading days
- 5 Years / 1260 trading days

Backtest output includes:

- Strategy metrics: total return, CAGR, Sharpe, max drawdown, and final value.
- Strategy vs benchmark comparison table.
- Equity curve.
- Drawdown chart.
- Rolling 63-day Sharpe chart.
- Rebalance history.
- Weight history.
- Turnover and transaction-cost table.
- Alpha/beta regression table.
- Fallback warnings when optimization fails and equal-weight fallback is used.
- Optional debug panel for inputs, data quality, rebalance logs, costs, returns, metrics, warnings, and cost logs.

## Monte Carlo Simulation

Monte Carlo simulation runs only after the selected portfolio passes the configured alpha and information-ratio gate.

The simulation uses the selected portfolio weights, historical mean returns, and covariance from the cleaned return data. It generates many possible future portfolio paths for the selected trading-day horizon and initial capital.

The dashboard shows:

- Expected final value: average ending value across simulated paths.
- Expected PnL: average profit or loss versus starting capital.
- Mean return: average simulated return over the selected horizon.
- Return standard deviation: dispersion of simulated returns.
- Simulated Sharpe ratio: mean simulated return divided by simulated return standard deviation for the selected horizon.
- Probability of gain: share of paths ending above starting capital.
- Probability of loss: share of paths ending below starting capital.
- Probability breakeven: share of paths ending effectively unchanged.
- Probability >= mean return: share of paths at or above the average return.
- Average gain if gain: average profit among winning paths only.
- Average loss if loss: average loss among losing paths only, shown as a positive number.
- Portfolio expected value: starting capital plus expected PnL, equal to average final portfolio value.
- VaR: loss threshold for the chosen confidence level.
- CVaR: average loss in the worst tail of simulated final outcomes.
- Max loss: worst final simulated loss across all paths. This is not an intra-period drawdown.

All Monte Carlo metric descriptions are shown as tooltip help icons in the UI to keep the dashboard clean.

## Optimization Methods

### Mean-Variance / Max Sharpe

Maximizes expected excess return per unit of volatility subject to long-only weights and the selected maximum-weight cap.

### Minimum Variance

Finds the lowest-volatility portfolio subject to the same weight constraints.

### Risk Parity

Targets a more balanced contribution to total portfolio risk from each asset.

### Equal Weight

Allocates equally across all selected assets. This is also used as a transparent fallback for failed rebalance periods in backtesting.

### Custom Manual Allocation

Lets you enter portfolio weights as percentages. The app validates ticker coverage, duplicates, numeric values, negative weights, and the sum of weights. If normalization is enabled, manual percentages are normalized into portfolio weights.

## Alpha Gate

The alpha gate is a validation layer before Monte Carlo output.

It compares the selected portfolio's realized historical returns with the benchmark and calculates:

- Alpha
- Beta
- Information ratio
- Tracking error
- Overlapping trading observations

The gate passes only when the selected portfolio clears the configured alpha and information-ratio thresholds. If it fails, the app still shows portfolio diagnostics, but skips Monte Carlo simulation so the simulated forward-risk section is not shown for a rejected strategy.

## How The Project Is Organized

```text
project/
|-- app.py
|-- portfolio_engine.py
|-- backtesting_engine.py
|-- requirements.txt
|-- run_app.ps1
`-- core/
    |-- __init__.py
    |-- constants.py
    |-- data_utils.py
    |-- metrics.py
    |-- optimizer.py
    `-- validation.py
```

### `app.py`

Streamlit UI only. It owns tabs, forms, charts, progress/status messages, metric formatting, and session-state wiring between the optimizer, comparison, and backtesting views.

### `portfolio_engine.py`

Current-period portfolio analysis orchestration. It resolves tickers, loads data, computes returns, runs optimizers, builds comparison tables, evaluates alpha, and prepares the data used by the Portfolio Optimization and Strategy Comparison tabs.

### `backtesting_engine.py`

Walk-forward backtesting orchestration. It handles rebalance schedules, trailing training windows, per-period optimization, cost application, comparison returns, regression output, fallback logs, and debug output.

### `core/constants.py`

Shared constants, defaults, strategy names, rebalance options, and common Indian stock/index name mappings.

### `core/data_utils.py`

Data loading, adjusted-close extraction, price cleaning, return calculation, data-quality summaries, and return-frame alignment.

### `core/metrics.py`

Shared financial calculations: returns, CAGR, volatility, Sharpe, Sortino, drawdowns, VaR, CVaR, turnover, tracking error, information ratio, alpha evaluation, rolling Sharpe, risk contribution, performance tables, and Monte Carlo risk metrics.

### `core/optimizer.py`

Optimization implementations and fallbacks. The optimizer tries the preferred optimizer path first and falls back to SciPy or random-search style solutions when necessary.

### `core/validation.py`

Input validation and normalization for tickers, dates, strategies, lookback windows, rebalance frequencies, weights, custom allocations, and finite data frames.

## Installation

Use Python 3.11+ if possible.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencies:

- `streamlit`
- `altair`
- `yfinance`
- `PyPortfolioOpt`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `pandas`
- `numpy`

## Run The App

From the project folder:

```powershell
streamlit run app.py
```

Or use the helper script:

```powershell
.\run_app.ps1
```

The helper script looks for `python` on PATH first. If it is not available, it tries the Codex bundled Python runtime at:

```text
%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

If Streamlit is missing for the selected Python interpreter, install requirements first:

```powershell
pip install -r requirements.txt
```

## Programmatic Backtest Example

```python
from backtesting_engine import run_walk_forward_backtest

result = run_walk_forward_backtest(
    tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    benchmark="^NSEI",
    start="2020-01-01",
    end="2026-04-30",
    strategy="Mean-Variance / Max Sharpe",
    rebalance_frequency="Monthly",
    lookback_days=504,
    initial_capital=100000,
    transaction_cost=0.002,
    slippage=0.001,
    risk_free_rate=0.06,
    max_weight=0.4,
    comparison_mode=True,
    debug=True,
)
```

## Important Notes

- The app uses daily simple returns for portfolio aggregation.
- Yahoo Finance data availability can vary by ticker and date range.
- A long lookback window requires enough historical data before the first out-of-sample period can be created.
- None-frequency backtests are static-hold simulations using supplied weights, not walk-forward validations.
- Walk-forward backtests intentionally avoid look-ahead bias by training only on data before each rebalance date.
- Custom manual backtests hold the entered static allocation for None frequency or re-apply it at each walk-forward rebalance.
- Optimizer fallback events are surfaced instead of silently hidden.
- Historical performance, optimized weights, and simulated results are decision-support outputs, not financial advice.
