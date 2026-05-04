# Modular Portfolio Dashboard With Backtesting

## Summary
Refactor the dashboard into the requested architecture, keeping `app.py` as Streamlit UI only and moving every financial calculation into shared core modules. The backtesting tab will run a strict walk-forward process where each training window ends before the rebalance date, uses the same optimizer interface as the portfolio tab, applies turnover-based costs on the first out-of-sample day, and exposes fallback/debug diagnostics instead of hiding failures.

## Final Folder Structure
```text
project/
├── app.py
├── portfolio_engine.py
├── backtesting_engine.py
├── requirements.txt
└── core/
    ├── __init__.py
    ├── constants.py
    ├── data_utils.py
    ├── metrics.py
    ├── optimizer.py
    └── validation.py
```

## Key Implementation Changes
- Move constants, strategy labels, defaults, ticker aliases, and supported rebalance/method names into `core/constants.py`.
- Move ticker/custom-weight validation, normalization, date checks, and numeric input checks into `core/validation.py`.
- Move Yahoo price fetching, adjusted-close extraction, price cleaning, simple return calculation, and intersection-date alignment into `core/data_utils.py`.
- Move all return, drawdown, CAGR, volatility, Sharpe, Sortino, VaR/CVaR, alpha/beta, information ratio, tracking error, rolling Sharpe, turnover, risk contribution, and summary metrics into `core/metrics.py`.
- Move all strategy weight generation into `core/optimizer.py` with public methods:
  `optimize_max_sharpe`, `optimize_min_variance`, `optimize_risk_parity`, `optimize_equal_weight`, `apply_custom_weights`.
- Replace hidden optimizer fallbacks with explicit `OptimizationResult` metadata:
  `optimizer_success`, `fallback_used`, `optimizer_name`, `message`, `weights`.
- Keep equal-weight fallback centralized in the optimizer interface and retry the requested optimizer at every rebalance.
- Refactor `portfolio_engine.py` into an orchestration layer for current-period optimization and strategy comparison only; no duplicated metric/data logic remains there.
- Add `backtesting_engine.py` with `run_walk_forward_backtest(...)`, returning a `BacktestResult` containing equity curve, drawdown, rolling Sharpe, benchmark comparison, weight history, rebalance table, cost table, metrics, fallback summary, and `debug_info`.

## Backtesting Behavior
- Rebalance dates are actual trading dates after enough prior lookback data exists.
- Training data uses `returns.index < rebalance_date`; out-of-sample returns start at `rebalance_date`.
- Each holding period runs from the rebalance date through the day before the next rebalance.
- Turnover is `0.5 * sum(abs(new_w - old_w))`.
- Cost rate is `transaction_cost + slippage`; cost is subtracted from the first out-of-sample return of that period.
- Custom weights are static allocations, validated once, then reapplied at each rebalance.
- Regime filter ships disabled by default. If enabled, it detects benchmark trend and volatility regimes; UI exposes behavior as `Skip rebalance` or `Move to cash`.

## App Integration
- `app.py` will create three tabs:
  `Portfolio Optimization`, `Strategy Comparison`, `Backtesting`.
- Shared inputs remain in Streamlit controls, but all computation goes through `portfolio_engine` or `backtesting_engine`.
- Backtesting tab inputs include tickers, dates, capital, strategy, rebalance frequency, lookback, transaction cost, slippage, risk-free rate, custom weights, regime options, and debug mode.
- Backtesting outputs include equity curve, drawdown, rolling Sharpe, weights history, rebalance table, benchmark comparison, fallback warning, fallback dates, and debug panel.
- Debug logs are capped to first 5 and last 5 rebalances by default.

## Requirements And Usage
- Keep existing dependencies: `streamlit`, `altair`, `yfinance`, `PyPortfolioOpt`, `scipy`, `scikit-learn`, `pandas`, `numpy`.
- Run:
```powershell
pip install -r requirements.txt
streamlit run app.py
```
- Engine example:
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
    debug=True,
)
```

## Test And Debug Checklists
- Compile: `python -m py_compile app.py portfolio_engine.py backtesting_engine.py core/*.py`.
- Smoke test with synthetic prices: verify weights sum to 1, no NaNs, no look-ahead, costs hit only first out-of-sample day, and fallback logs reset each rebalance.
- Validate metrics parity: portfolio tab and backtest use identical simple-return definitions and identical metric functions.
- UI test: run each tab, custom weights, optimizer failure fallback, missing ticker, insufficient lookback, high cost, and debug enabled.
- Debug panel must show inputs, data quality, rebalance logs, cost logs, return checks, metric checks, warnings, fallback count, and fallback dates.

## Failure Cases And Fixes
- Missing packages: install `requirements.txt`; current local `.venv` appears broken, so recreate it if `pip` reports Python access errors.
- Bad ticker or no data: show dropped/missing ticker diagnostics and stop if fewer than two valid assets remain.
- Insufficient history: require enough aligned return rows for the selected lookback before first rebalance.
- Optimizer failure or singular covariance: use equal-weight only for that rebalance, log `fallback_used=True`, retry next rebalance.
- Suspicious outputs: warn for NaNs, extreme daily returns, high turnover, high drawdown, or implausible Sharpe.
