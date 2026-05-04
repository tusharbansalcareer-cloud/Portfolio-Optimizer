# Knowledge Base: Professional Backtesting Engine for Swing-Trading Portfolio Dashboard

## 0. Purpose

This knowledge source defines the theory, assumptions, formulas, architecture, and implementation plan for a **dashboard-based backtesting engine**.

The dashboard has two main pages/tabs:

1. **Portfolio Optimization Tab**
   - User inputs 4–6 stocks.
   - System optimizes portfolio weights using maximum Sharpe ratio.

2. **Backtesting Engine Tab**
   - User tests whether the optimized portfolio logic is reliable historically.
   - System performs walk-forward backtesting, transaction-cost adjustment, benchmark comparison, regime analysis, and validation.
   - Only strategies that pass validation should be considered for Monte Carlo simulation or real-world decision support.

This document is intended to be fed into a coding agent such as GPT-5.5 inside VS Code.

---

## 1. Core Objective

Build a **production-ready backtesting engine** for a small swing-trading portfolio of approximately 4–6 stocks.

The engine must answer:

> If I had used this optimization process historically, would it have produced reliable, risk-adjusted, benchmark-beating returns after realistic costs?

The goal is not just to generate attractive backtest charts. The goal is to reduce false confidence and create a system that can support daily investment decision-making.

---

## 2. Strategy Context

### 2.1 Strategy Type

- Style: Portfolio-based swing trading
- Universe: User-defined stock list
- Typical number of stocks: 4–6
- Holding period: days to weeks
- Rebalancing: weekly, biweekly, or monthly
- Optimization objective: maximum Sharpe ratio
- Benchmark default: NIFTY 50 (`^NSEI`)
- Data source default: Yahoo Finance via `yfinance`
- Dashboard framework: Streamlit preferred

### 2.2 What the Strategy Is

The strategy does not forecast individual stock prices directly.

Instead, it:

1. Takes a basket of candidate stocks.
2. Estimates their expected returns and covariance from historical data.
3. Optimizes portfolio weights for best expected risk-adjusted return.
4. Tests those weights out-of-sample through walk-forward backtesting.
5. Compares the strategy against relevant benchmarks.
6. Applies transaction costs and slippage.
7. Checks if performance survives realistic assumptions.

### 2.3 What the Strategy Is Not

It is not:

- A guaranteed alpha engine
- A pure momentum strategy
- A pure mean-reversion strategy
- A buy-and-hold portfolio allocator
- A high-frequency trading model
- A standalone investment decision system without human review

---

## 3. Required Dashboard Structure

### 3.1 Tab 1: Portfolio Optimization

Purpose:
- Optimize current portfolio weights.

Inputs:
- Tickers
- Start date
- End date
- Risk-free rate
- Weight constraints
- Benchmark
- Optimization method

Outputs:
- Optimal weights
- Expected annual return
- Annual volatility
- Sharpe ratio
- Correlation matrix
- Optional efficient frontier

### 3.2 Tab 2: Backtesting Engine

Purpose:
- Test whether the optimization process works historically.

Inputs:
- Tickers
- Benchmark
- Start date
- End date
- Lookback window
- Rebalance frequency
- Transaction cost
- Slippage
- Risk-free rate
- Max weight per asset
- Min weight per asset
- Regime detection toggle
- Validation thresholds

Outputs:
- Equity curve
- Benchmark comparison
- Drawdown curve
- Rolling Sharpe
- Rolling alpha
- Weight history
- Turnover history
- Cost drag
- Regime performance table
- Final pass/fail verdict

---

## 4. Key Red Flags to Avoid

### 4.1 Look-Ahead Bias

Never use future data to make past decisions.

Wrong:
- Optimize weights using the full dataset, then backtest on that same dataset.

Correct:
- At each rebalance date, only use data available before that date.

### 4.2 Survivorship Bias

If the stock universe only contains current winners, the backtest is biased.

For user-selected stocks, document this limitation:
- The system tests only the selected securities.
- It does not reconstruct the historically investable universe.

### 4.3 Overfitting

Optimization can fit noise.

Common causes:
- Too many assets
- Too short a lookback window
- Too frequent rebalancing
- No transaction costs
- No out-of-sample testing
- Optimizing multiple parameter sets and only keeping the best

### 4.4 Ignoring Costs

Gross returns are not sufficient.

Must include:
- Brokerage
- STT / exchange-related charges where applicable
- Slippage
- Bid-ask spread proxy
- Taxes/fees proxy
- Turnover-based costs

### 4.5 Benchmark Mismatch

Comparing against only NIFTY 50 may be insufficient.

Use:
- NIFTY 50 benchmark
- Equal-weight version of selected stocks
- Optional sector index if relevant
- Optional cash or risk-free benchmark

### 4.6 Monte Carlo Misuse

Monte Carlo should not be used before historical validation.

Correct sequence:

```text
Optimize → Backtest → Validate → Simulate
```

If backtest fails, forward simulation should be disabled or marked invalid.

---

## 5. Theoretical Assumptions

### 5.1 Return Assumptions

Most portfolio optimization assumes returns are estimated from history.

Daily simple return:

```text
r_t = (P_t / P_{t-1}) - 1
```

Log return:

```text
r_t = ln(P_t / P_{t-1})
```

Default recommendation:
- Use simple returns for portfolio aggregation.
- Log returns may be used for statistical analysis, but portfolio return should be calculated using weighted simple returns.

### 5.2 Expected Return Assumption

Historical average return is a noisy estimator.

Mean daily return:

```text
μ_i = (1 / N) * Σ r_{i,t}
```

Annualized arithmetic return approximation:

```text
μ_annual = μ_daily * 252
```

Geometric annual return:

```text
CAGR = (Ending Value / Beginning Value)^(252 / N) - 1
```

For swing trading:
- Avoid overtrusting long-term expected returns.
- Consider shorter rolling windows, but not so short that estimates become unstable.

### 5.3 Covariance Assumption

Covariance matrix estimates asset co-movement.

Covariance:

```text
Σ_ij = Cov(r_i, r_j)
```

Portfolio variance:

```text
σ_p² = wᵀ Σ w
```

Annualized portfolio volatility:

```text
σ_p,annual = sqrt(wᵀ Σ_daily w) * sqrt(252)
```

### 5.4 Normality Assumption

Many models assume returns are normally distributed.

Real-world returns often have:
- Fat tails
- Skewness
- Volatility clustering
- Regime shifts

Therefore:
- Historical backtesting matters more than theoretical Gaussian assumptions.
- VaR/CVaR should ideally be calculated using historical or simulation-based outcomes.

---

## 6. Portfolio Optimization Formulas

### 6.1 Portfolio Return

For weight vector `w` and asset return vector `r_t`:

```text
r_p,t = Σ(w_i * r_i,t)
```

Vector form:

```text
r_p,t = wᵀ r_t
```

### 6.2 Portfolio Expected Return

```text
E[R_p] = wᵀ μ
```

Annualized:

```text
E[R_p,annual] = wᵀ μ_daily * 252
```

### 6.3 Portfolio Volatility

```text
σ_p = sqrt(wᵀ Σ w)
```

Annualized:

```text
σ_p,annual = sqrt(wᵀ Σ_daily w) * sqrt(252)
```

### 6.4 Sharpe Ratio

```text
Sharpe = (R_p - R_f) / σ_p
```

Annualized:

```text
Sharpe_annual = (R_p,annual - R_f,annual) / σ_p,annual
```

Where:
- `R_p` = portfolio return
- `R_f` = risk-free rate
- `σ_p` = portfolio volatility

### 6.5 Maximum Sharpe Optimization

Objective:

```text
maximize: (wᵀ μ - R_f) / sqrt(wᵀ Σ w)
```

Subject to:

```text
Σ w_i = 1
w_i >= min_weight
w_i <= max_weight
```

Default practical constraints:
- Long-only weights
- Min weight: 0
- Max weight per stock: 40% or 50%
- Fully invested: sum of weights = 1

For 4–6 stocks, avoid allowing 100% concentration unless explicitly intended.

---

## 7. Backtesting Methodology

## 7.1 Preferred Method: Walk-Forward Backtest

Walk-forward testing is the core professional method.

At each rebalance date:

1. Select historical lookback window.
2. Estimate expected returns and covariance.
3. Optimize weights using only historical data.
4. Apply those weights over the next holding period.
5. Record portfolio returns.
6. Move to next rebalance date.
7. Repeat until end of test period.

### 7.2 Walk-Forward Timeline Example

```text
Lookback: 504 trading days
Rebalance: monthly

Window 1:
Train: Day 1 to Day 504
Hold: Day 505 to next rebalance

Window 2:
Train: Day 22 to Day 525
Hold: Day 526 to next rebalance

Continue until end date.
```

### 7.3 Recommended Defaults

| Parameter | Default | Reason |
|---|---:|---|
| Lookback window | 504 trading days | Balances stability and recency |
| Alternative lookback | 252 / 756 days | Sensitivity testing |
| Rebalance frequency | Monthly | Reduces cost drag |
| Swing-trading alternative | Weekly | More responsive, more costs |
| Max asset weight | 40% | Prevents concentration |
| Min asset weight | 0% | Long-only |
| Risk-free rate | User input, default 0 | Simpler unless accurate source is used |
| Transaction cost | 0.10% to 0.30% per turnover | Practical placeholder |
| Slippage | 0.05% to 0.25% | Depends on liquidity |
| Benchmark | `^NSEI` | NIFTY 50 |

---

## 8. Transaction Cost Model

### 8.1 Turnover

Turnover at rebalance:

```text
Turnover_t = Σ |w_new,i - w_old,i|
```

If moving from cash or no position:

```text
w_old = 0
```

### 8.2 Cost

```text
Cost_t = Turnover_t * cost_rate
```

Where `cost_rate` includes:
- Brokerage
- Slippage
- Bid-ask spread
- Taxes/fees proxy

### 8.3 Net Return

If cost is applied on rebalance day:

```text
r_net,t = r_gross,t - Cost_t
```

For more realistic implementation:

```text
portfolio_value_after_cost = portfolio_value_before_cost * (1 - Cost_t)
```

Then apply market return.

### 8.4 Cost Drag

Total cost drag:

```text
Cost Drag = Gross CAGR - Net CAGR
```

Also track:
- Total turnover
- Average turnover per rebalance
- Number of rebalances
- Total cost paid

### 8.5 Practical Guidance

High turnover is dangerous.

A strategy that beats the benchmark before costs but fails after costs should be rejected.

---

## 9. Benchmarking and Alpha

### 9.1 Benchmark Return

Benchmark daily return:

```text
r_b,t = (B_t / B_{t-1}) - 1
```

### 9.2 Excess Return

```text
Excess Return_t = r_p,t - r_b,t
```

Annualized excess return:

```text
Excess Return_annual = mean(r_p,t - r_b,t) * 252
```

### 9.3 CAPM Regression

Regression model:

```text
r_p,t - r_f,t = α + β * (r_b,t - r_f,t) + ε_t
```

If risk-free rate is omitted for simplicity:

```text
r_p,t = α + β * r_b,t + ε_t
```

Where:
- `α` = daily alpha
- `β` = market sensitivity
- `ε_t` = residual return

Annualized alpha:

```text
Alpha_annual = α_daily * 252
```

### 9.4 Beta

```text
β = Cov(r_p, r_b) / Var(r_b)
```

Interpretation:
- Beta > 1: portfolio more volatile than benchmark
- Beta < 1: portfolio less sensitive to benchmark
- Beta near 0: low market exposure

### 9.5 Alpha Significance

Alpha alone is not enough.

Use regression p-value:

```text
alpha_p_value < 0.05
```

This indicates statistical significance at the 5% level.

However, for small samples and noisy markets, also inspect:
- Stability of rolling alpha
- Performance after costs
- Robustness across parameter settings

### 9.6 Information Ratio

```text
IR = Annualized Excess Return / Annualized Tracking Error
```

Tracking error:

```text
TE = std(r_p,t - r_b,t) * sqrt(252)
```

Information ratio:

```text
IR = mean(r_p,t - r_b,t) * 252 / (std(r_p,t - r_b,t) * sqrt(252))
```

Practical interpretation:
- IR < 0: underperformance
- IR 0.0–0.3: weak
- IR 0.3–0.5: acceptable but not strong
- IR > 0.5: good
- IR > 1.0: very strong, verify carefully

---

## 10. Performance Metrics

### 10.1 Equity Curve

Starting capital:

```text
V_0 = initial_capital
```

Portfolio value:

```text
V_t = V_{t-1} * (1 + r_p,t)
```

### 10.2 CAGR

```text
CAGR = (V_T / V_0)^(252 / N) - 1
```

Where:
- `V_T` = ending value
- `V_0` = starting value
- `N` = number of trading days

### 10.3 Annualized Volatility

```text
Volatility = std(r_p,t) * sqrt(252)
```

### 10.4 Sharpe Ratio

```text
Sharpe = (mean(r_p,t) * 252 - R_f) / (std(r_p,t) * sqrt(252))
```

### 10.5 Sortino Ratio

Uses downside deviation instead of total volatility.

Downside returns:

```text
r_down,t = min(0, r_p,t - MAR)
```

Downside deviation:

```text
Downside Deviation = std(r_down,t) * sqrt(252)
```

Sortino:

```text
Sortino = (R_p,annual - MAR_annual) / Downside Deviation
```

Default MAR:
- 0
- or risk-free rate

### 10.6 Drawdown

Running peak:

```text
Peak_t = max(V_0, V_1, ..., V_t)
```

Drawdown:

```text
DD_t = (V_t / Peak_t) - 1
```

Maximum drawdown:

```text
MaxDD = min(DD_t)
```

### 10.7 Calmar Ratio

```text
Calmar = CAGR / abs(Max Drawdown)
```

### 10.8 Win Rate

```text
Win Rate = Number of positive return periods / Total periods
```

Can be calculated daily, weekly, or per rebalance period.

### 10.9 Profit Factor

```text
Profit Factor = Sum of positive returns / abs(Sum of negative returns)
```

### 10.10 VaR

Historical VaR at confidence level `q`:

```text
VaR_q = percentile(returns, 1 - q)
```

For 95% confidence:

```text
VaR_95 = percentile(returns, 5)
```

### 10.11 CVaR / Expected Shortfall

```text
CVaR_95 = mean(returns where returns <= VaR_95)
```

CVaR is usually more informative than VaR because it measures average tail loss.

---

## 11. Rolling Metrics

### 11.1 Rolling Sharpe

```text
Rolling Sharpe_t = mean(r_{t-window:t}) * 252 / (std(r_{t-window:t}) * sqrt(252))
```

Recommended windows:
- 63 trading days
- 126 trading days
- 252 trading days

### 11.2 Rolling Alpha

Run CAPM regression over rolling window.

```text
r_p = α + β r_b + ε
```

Track:
- Rolling alpha
- Rolling beta
- Rolling alpha p-value if feasible

### 11.3 Rolling Drawdown

Track ongoing drawdown from recent peak.

Used to detect live deterioration.

---

## 12. Regime Detection

Regime analysis answers:

> Does the strategy only work in certain market environments?

### 12.1 Volatility Regime

Benchmark rolling volatility:

```text
Vol_t = std(r_b,t-window:t) * sqrt(252)
```

Define:
- Low volatility: below 40th percentile
- Medium volatility: 40th to 70th percentile
- High volatility: above 70th percentile

Alternative:
- High vol if rolling volatility > long-term rolling volatility median

### 12.2 Trend Regime

Use moving average filter on benchmark.

```text
Trend_t = Price_t > SMA_200_t
```

Regimes:
- Bull/trending: benchmark above 200-day moving average
- Bear/weak: benchmark below 200-day moving average

For swing trading, also consider:
- 50-day SMA
- 100-day SMA
- 200-day SMA

### 12.3 Combined Regimes

Possible labels:
- Bull + Low Vol
- Bull + High Vol
- Bear + Low Vol
- Bear + High Vol

### 12.4 Regime Performance

For each regime calculate:
- CAGR
- Sharpe
- Max drawdown
- Hit rate
- Average daily return
- Number of observations

If performance only works in one regime, the dashboard should make that obvious.

---

## 13. Validation and Decision Rules

### 13.1 Strategy Pass/Fail Gate

A strategy should pass only if it meets minimum criteria.

Recommended default gate:

```text
Net Sharpe > 1.0
Net CAGR > benchmark CAGR
Alpha annualized > 0
Alpha p-value < 0.10 or < 0.05
Information Ratio > 0.3
Max Drawdown > -25%
Cost-adjusted returns remain positive
```

For stricter professional standard:

```text
Net Sharpe > 1.25
Information Ratio > 0.5
Alpha p-value < 0.05
Max Drawdown > -20%
Outperforms equal-weight basket
```

### 13.2 Dashboard Verdict

Possible verdicts:

```text
PASS
WATCHLIST
FAIL
```

PASS:
- Meets all core thresholds.

WATCHLIST:
- Good returns but weak significance or high drawdown.

FAIL:
- Underperforms benchmark or fails after costs.

### 13.3 Avoiding False Positives

Use:
- Walk-forward backtest
- Multiple lookback windows
- Multiple rebalance frequencies
- Transaction costs
- Equal-weight benchmark
- Regime segmentation
- Rolling performance checks

Avoid:
- Optimizing thresholds repeatedly until results look good
- Ignoring failed parameter sets
- Only reporting the best result

---

## 14. Robustness Testing

### 14.1 Parameter Sensitivity

Test combinations:

Lookback windows:
- 252
- 504
- 756

Rebalance:
- Weekly
- Biweekly
- Monthly

Max weight:
- 30%
- 40%
- 50%

Cost assumptions:
- Low cost: 0.10%
- Base cost: 0.20%
- Stress cost: 0.40%

### 14.2 Robustness Rule

A strategy is more reliable if it works across nearby parameter values.

Bad sign:
- Only one exact parameter setting works.

Good sign:
- Similar results across multiple reasonable settings.

### 14.3 Stress Scenarios

Test:
- High-cost scenario
- High-volatility periods
- Market drawdowns
- Sudden benchmark declines
- Missing data for one asset
- One stock removed from basket

---

## 15. Practical Implementation Architecture

### 15.1 Recommended Python Libraries

Core:
- `pandas`
- `numpy`
- `yfinance`
- `scipy`
- `statsmodels`
- `PyPortfolioOpt`

Visualization:
- `plotly`
- `matplotlib`

Dashboard:
- `streamlit`

Optional:
- `scikit-learn` for regime classification
- `hmmlearn` only if advanced Hidden Markov Models are desired

### 15.2 Module Structure

Suggested file structure:

```text
portfolio_dashboard/
│
├── app.py
├── requirements.txt
│
├── pages/
│   ├── 1_Optimization.py
│   └── 2_Backtesting.py
│
├── src/
│   ├── data.py
│   ├── optimization.py
│   ├── backtest.py
│   ├── costs.py
│   ├── metrics.py
│   ├── benchmark.py
│   ├── regimes.py
│   ├── validation.py
│   └── charts.py
```

---

## 16. Required Functions

### 16.1 Data Functions

```python
def fetch_prices(tickers, start, end):
    # Download adjusted close prices.
    # Return clean price DataFrame.
    pass
```

```python
def compute_returns(prices):
    # Compute daily simple returns.
    # Drop rows where all values are NaN.
    # Handle missing data carefully.
    pass
```

### 16.2 Optimization Function

```python
def optimize_max_sharpe(returns, risk_free_rate=0.0, min_weight=0.0, max_weight=0.4):
    # Estimate expected returns and covariance.
    # Optimize for maximum Sharpe ratio.
    # Return clean weights as dict or Series.
    pass
```

Implementation notes:
- Use PyPortfolioOpt.
- Use constraints.
- If optimization fails, fallback to equal weight.
- Normalize weights.

### 16.3 Walk-Forward Backtest Function

```python
def walk_forward_backtest(
    prices,
    benchmark_prices,
    lookback_days=504,
    rebalance_frequency="M",
    transaction_cost=0.002,
    slippage=0.001,
    max_weight=0.4,
    risk_free_rate=0.0
):
    # Run walk-forward optimization and backtest.
    # Return portfolio returns, equity curve, weights, turnover, costs, and diagnostics.
    pass
```

### 16.4 Transaction Cost Function

```python
def calculate_turnover(old_weights, new_weights):
    # Sum absolute changes in portfolio weights.
    pass
```

```python
def apply_transaction_cost(portfolio_return, turnover, cost_rate):
    # Adjust gross portfolio return for transaction costs.
    pass
```

### 16.5 Metrics Function

```python
def calculate_performance_metrics(portfolio_returns, benchmark_returns=None, risk_free_rate=0.0):
    # Calculate CAGR, volatility, Sharpe, Sortino, max drawdown,
    # alpha, beta, information ratio, VaR, CVaR.
    pass
```

### 16.6 Regime Function

```python
def detect_regimes(benchmark_prices, benchmark_returns):
    # Create trend and volatility regime labels.
    pass
```

### 16.7 Validation Function

```python
def validate_strategy(metrics, thresholds):
    # Return PASS, WATCHLIST, or FAIL with reasons.
    pass
```

---

## 17. Walk-Forward Pseudocode

```python
prices = fetch_prices(tickers, start, end)
returns = compute_returns(prices)

benchmark_prices = fetch_prices([benchmark], start, end)
benchmark_returns = compute_returns(benchmark_prices)

rebalance_dates = generate_rebalance_dates(returns.index, frequency)

old_weights = zero_weights
portfolio_returns = []
weight_history = []
turnover_history = []
cost_history = []

for date in rebalance_dates:

    train_start = date - lookback_window
    train_returns = returns.loc[train_start:date]

    if len(train_returns) < lookback_days:
        continue

    new_weights = optimize_max_sharpe(train_returns)

    next_period_returns = returns.loc[date:next_rebalance_date]

    turnover = calculate_turnover(old_weights, new_weights)
    cost_rate = transaction_cost + slippage
    cost = turnover * cost_rate

    for day in next_period_returns.index:
        gross_return = dot(new_weights, next_period_returns.loc[day])
        net_return = gross_return

        if day == first_day_of_period:
            net_return = gross_return - cost

        portfolio_returns.append(net_return)

    store weights, turnover, cost

    old_weights = new_weights
```

Important:
- Do not include the rebalance day return if weights were not known before market close.
- Prefer applying new weights from the next trading day after rebalance.
- Avoid accidental look-ahead.

---

## 18. Data Handling Rules

### 18.1 Missing Data

Options:
1. Drop rows with missing values.
2. Forward-fill prices before computing returns.
3. Exclude tickers with insufficient history.

Recommended:
- Forward-fill minor missing price gaps.
- Drop remaining NaNs.
- Warn user if a ticker has poor data coverage.

### 18.2 Minimum Data Requirement

For each ticker:
- Require at least 80% data availability in selected period.
- Require enough data for lookback window.

### 18.3 Corporate Actions

Use adjusted close prices to account for:
- Splits
- Dividends
- Bonus issues when available

### 18.4 Data Quality Warning

Yahoo Finance data can have errors.

Dashboard should display:
- Last download timestamp
- Missing data count
- Tickers removed due to insufficient data

---

## 19. Practical Defaults for Indian Equity Swing Trading

### 19.1 Default Dashboard Inputs

```text
Tickers: user input
Benchmark: ^NSEI
Lookback: 504 trading days
Rebalance: Monthly
Max weight: 40%
Min weight: 0%
Transaction cost: 0.20%
Slippage: 0.10%
Initial capital: 100000
Risk-free rate: 0%
```

### 19.2 Alternative Swing Trading Defaults

More active:

```text
Lookback: 252 trading days
Rebalance: Weekly
Max weight: 35%
Transaction cost + slippage: 0.30% to 0.50%
```

More stable:

```text
Lookback: 756 trading days
Rebalance: Monthly
Max weight: 40%
Transaction cost + slippage: 0.20%
```

---

## 20. Dashboard Layout for Backtesting Tab

### 20.1 Sidebar Inputs

Sections:

1. Universe
   - Tickers
   - Benchmark
   - Date range

2. Backtest Settings
   - Lookback window
   - Rebalance frequency
   - Initial capital

3. Optimization Settings
   - Risk-free rate
   - Max weight
   - Min weight

4. Cost Settings
   - Transaction cost
   - Slippage

5. Validation Settings
   - Minimum Sharpe
   - Minimum information ratio
   - Max drawdown
   - Alpha p-value threshold

6. Regime Settings
   - Enable/disable
   - Trend window
   - Volatility window

### 20.2 Main Page Outputs

Top summary cards:
- Final portfolio value
- CAGR
- Sharpe
- Max drawdown
- Alpha
- Information ratio
- Verdict

Charts:
- Equity curve vs benchmark
- Drawdown curve
- Rolling Sharpe
- Weight allocation over time
- Turnover over time
- Regime performance chart/table

Tables:
- Backtest metrics
- Benchmark metrics
- Weight history
- Rebalance log
- Cost summary
- Validation reasons

---

## 21. Rebalance Logic Details

### 21.1 Rebalance Frequency Options

- Weekly: more responsive, higher costs
- Biweekly: balanced
- Monthly: lower costs, more stable
- Quarterly: may be too slow for swing trading

Recommended default:
- Monthly for first version
- Weekly as optional mode

### 21.2 Date Alignment

Use actual trading dates.

If scheduled rebalance date is not a trading day:
- Use next available trading day
- Or previous available trading day
- Be consistent and document choice

Recommended:
- Use the next available trading day and apply weights from the following trading day.

### 21.3 Preventing Look-Ahead

If optimization is performed using data through date `t`, weights should apply from `t+1`, not from `t`.

---

## 22. Equal-Weight Baseline

For selected tickers, compute equal-weight portfolio.

```text
w_i = 1 / N
```

Equal-weight return:

```text
r_eq,t = mean(r_i,t)
```

Why:
- If optimized strategy cannot beat equal weight after costs, optimization may not add value.

---

## 23. Weight Constraints and Practical Risk Controls

### 23.1 Max Weight

Avoid excessive concentration.

Recommended:
- 40% max weight for 4–6 stocks
- 50% max weight only if user accepts concentration

### 23.2 Min Weight

Default:
- 0%

Avoid tiny weights:
- Clean weights below 1% to 0
- Renormalize

### 23.3 Long-Only Assumption

Default:
- No shorting

Reason:
- Most retail/investment swing-trading workflows are long-only.
- Shorting introduces borrow cost, margin, and execution complexity.

### 23.4 Cash Allocation

Optional:
- Allow cash if market regime is unfavorable.
- This requires modifying optimization constraints.

Initial version:
- Fully invested.

Advanced version:
- Add cash proxy or risk-free asset.

---

## 24. Error Handling

### 24.1 Optimization Failure

If optimizer fails:
- Use equal-weight fallback.
- Record warning.
- Do not silently fail.

### 24.2 Singular Covariance Matrix

Can happen with:
- Too few observations
- Highly correlated assets
- Missing data

Solutions:
- Use shrinkage covariance
- Add small diagonal regularization
- Use equal-weight fallback

### 24.3 Bad Tickers

If ticker download fails:
- Show invalid ticker list
- Remove failed tickers only with user warning
- Do not continue silently

---

## 25. Statistical Testing Notes

### 25.1 Alpha p-value

Use `statsmodels.OLS`.

Model:

```python
X = sm.add_constant(benchmark_returns)
model = sm.OLS(portfolio_returns, X).fit()
alpha = model.params["const"]
beta = model.params[benchmark_col]
alpha_p_value = model.pvalues["const"]
```

Annualized alpha:

```python
alpha_annual = alpha * 252
```

### 25.2 T-Statistic

```text
t_alpha = alpha / SE(alpha)
```

### 25.3 Interpretation

Even statistically significant alpha can decay live.

Use p-values as one filter, not final truth.

---

## 26. Professional Practical Checks

Before trusting any result, check:

1. Did it beat NIFTY 50 after costs?
2. Did it beat equal weight after costs?
3. Was Sharpe above threshold?
4. Was alpha positive?
5. Was alpha statistically meaningful?
6. Was drawdown acceptable?
7. Did it work across regimes?
8. Did it work across nearby parameter values?
9. Was turnover reasonable?
10. Did one stock dominate results?

---

## 27. Recommended Version Roadmap

### Version 1: Reliable Basic Backtester

Implement:
- Walk-forward backtest
- Max Sharpe optimization
- NIFTY benchmark
- Equal-weight benchmark
- Costs and slippage
- Main metrics
- Validation verdict

### Version 2: Professional Enhancements

Add:
- Regime detection
- Rolling metrics
- Robustness grid
- Weight history visualizations
- Cost drag analysis

### Version 3: Advanced Institutional Features

Add:
- Covariance shrinkage
- Risk model comparison
- Factor exposure
- Cash allocation
- Stop-loss rules
- Volatility targeting
- Trade-level execution simulation

---

## 28. Implementation Priorities

Build in this order:

1. Data ingestion
2. Return calculation
3. Max Sharpe optimizer
4. Walk-forward rebalance loop
5. Transaction cost adjustment
6. Benchmark comparison
7. Metrics engine
8. Validation engine
9. Charts
10. Regime detection
11. Robustness testing

Do not build Monte Carlo before the backtest engine is reliable.

---

## 29. Final Required Output from Coding Agent

The coding agent should produce:

1. Streamlit dashboard with:
   - Optimization tab
   - Backtesting tab

2. Modular source code:
   - `data.py`
   - `optimization.py`
   - `backtest.py`
   - `metrics.py`
   - `regimes.py`
   - `validation.py`
   - `charts.py`

3. Backtesting tab outputs:
   - Equity curve
   - Drawdown curve
   - Metrics table
   - Alpha/beta table
   - Weight history
   - Turnover and cost table
   - Regime performance
   - Final pass/fail verdict

4. Safety features:
   - Data quality warnings
   - Failed ticker handling
   - Optimizer fallback
   - Clear assumptions displayed in UI

---

## 30. Final Design Principle

The backtesting tab must be built to disprove the strategy, not to market it.

A strategy is useful only if it survives:
- Out-of-sample testing
- Costs
- Benchmarks
- Regime shifts
- Parameter sensitivity
- Drawdowns

The dashboard should help the user avoid false confidence and make more disciplined investment decisions.

---

## 31. Legal and Risk Note

This system is a decision-support tool, not financial advice.

It should display a clear reminder:
- Historical performance does not guarantee future performance.
- Backtests can be wrong due to data quality, changing regimes, liquidity, and execution differences.
- User is responsible for investment decisions.
