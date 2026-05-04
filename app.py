from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from backtesting_engine import run_walk_forward_backtest
from core.constants import (
    DEFAULT_BENCHMARK,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SLIPPAGE,
    DEFAULT_TRANSACTION_COST,
    OPTIMIZATION_STRATEGIES,
    REBALANCE_MONTHLY,
    REBALANCE_NONE,
    STRATEGY_CUSTOM_MANUAL,
    SUPPORTED_REBALANCE_FREQUENCIES,
)
from core.validation import PortfolioError, normalize_tickers
from portfolio_engine import (
    PortfolioAnalysisResult,
    calculate_risk_metrics,
    get_portfolio_correlation_matrix,
    monte_carlo_display_values,
    monte_carlo_simulation,
    run_portfolio_analysis,
)


st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


DEFAULT_TICKER_TEXT = "reliance, tcs, hdfc bank, infosys, icici bank"

LOOKBACK_LABELS = {
    126: "6 Months / 126 trading days",
    252: "1 Year / 252 trading days",
    504: "2 Years / 504 trading days",
    756: "3 Years / 756 trading days",
    1260: "5 Years / 1260 trading days",
}

TAB_DESCRIPTIONS = {
    "Portfolio Optimization": (
        "Build an optimized long-only portfolio from selected tickers, evaluate it against the benchmark, "
        "and review allocation, risk contribution, correlation, drawdowns, and Monte Carlo risk."
    ),
    "Strategy Comparison": (
        "Compare Max Sharpe, Minimum Variance, Risk Parity, Equal Weight, and any enabled custom allocation "
        "on the same cleaned return history."
    ),
    "Backtesting": (
        "Run a static hold of the latest optimized weights or a walk-forward validation that retrains at each rebalance, "
        "then compare realized performance against benchmark and strategy baselines."
    ),
}

UI_HELP = {
    "tickers": f"Enter stock tickers separated by commas (e.g., RELIANCE.NS, TCS.NS). Unit: comma-separated names or Yahoo tickers. Default: {DEFAULT_TICKER_TEXT}.",
    "start_date": "Beginning of historical data used for analysis. Unit: calendar date.",
    "end_date": "End of historical data period. Unit: calendar date.",
    "benchmark": f"Market index used for comparison. Unit: Yahoo ticker or supported name. Default: {DEFAULT_BENCHMARK}.",
    "strategy": "Optimization method used to generate portfolio weights. Unit: strategy name.",
    "risk_free_rate": f"Annual risk-free return used in Sharpe ratio (e.g., 0.06 = 6%). Unit: annual decimal return. Default: {DEFAULT_RISK_FREE_RATE:.2f}.",
    "max_weight": f"Maximum allocation allowed per stock to avoid concentration risk. Unit: portfolio fraction. Default: {DEFAULT_MAX_WEIGHT:.2f}.",
    "l2_reg": "Penalty applied to large weights. Higher values increase diversification. Unit: penalty coefficient. Default: 0.00.",
    "alpha_threshold": "Minimum excess return over benchmark required for strategy acceptance. Unit: annual decimal return. Default: 0.00.",
    "information_ratio_threshold": "Measures consistency of outperformance vs benchmark. Unit: ratio. Default: 0.50.",
    "simulations": "Number of Monte Carlo portfolios generated. Higher = slower. Unit: simulation count. Default: 20000.",
    "time_horizon": "Holding period in days used for estimation. Unit: trading days. Default: 60.",
    "confidence_level": "Used for risk metrics like VaR (e.g., 0.95 = 95%). Unit: probability. Default: 0.95.",
    "random_seed": "Ensures reproducibility of results. Unit: integer seed. Default: 42.",
    "initial_portfolio": f"Starting investment amount. Unit: portfolio currency. Default: {int(DEFAULT_INITIAL_CAPITAL)}.",
    "enable_custom_weights": "Manually assign portfolio weights instead of optimization. Unit: on/off. Default: off.",
    "normalize_weights": "Ensure weights sum to 100%. Unit: on/off. Default: on.",
    "rebalance_frequency": "How often the backtest re-optimizes. None holds supplied weights through the full test. Unit: frequency. Default: Monthly.",
    "lookback_window": f"Trailing training window used before each rebalance. Ignored for None. Unit: trading days. Default: {DEFAULT_LOOKBACK_DAYS}.",
    "transaction_cost": f"Per-turnover transaction cost applied at rebalance. Unit: percent. Default: {DEFAULT_TRANSACTION_COST * 100:.2f}%.",
    "slippage": f"Estimated execution slippage applied at rebalance. Unit: basis points. Default: {DEFAULT_SLIPPAGE * 10000:.0f} bps.",
    "comparison_mode": "Include benchmark, equal-weight basket, and all optimizer strategies in the backtest comparison. Unit: on/off. Default: on.",
    "debug_mode": "Expose structured inputs, data quality, rebalance, cost, return, metric, and warning diagnostics. Unit: on/off. Default: off.",
}


def pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2%}"


def money(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.0f}"


def ratio(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.2f}"


def manual_weights_template(tickers: list[str]) -> pd.DataFrame:
    weight = 100.0 / len(tickers) if tickers else 0.0
    return pd.DataFrame({"Ticker": tickers, "Custom Weight %": [weight] * len(tickers)})


def weights_display_frame(weights: pd.Series) -> pd.DataFrame:
    display = weights.rename("Weight").reset_index()
    display = display.rename(columns={display.columns[0]: "Ticker"})
    display = display.sort_values("Weight", ascending=False)
    display["Weight"] = display["Weight"].map(pct)
    return display


def date_range_label(index: pd.Index) -> str:
    if len(index) == 0:
        return ""
    start_label = pd.Timestamp(index[0]).strftime("%d %b '%y")
    end_label = pd.Timestamp(index[-1]).strftime("%d %b '%y")
    return f"{start_label} - {end_label}"


def estimated_business_days(start_date: date, end_date: date) -> int:
    return len(pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date)))


def backtest_prefill_defaults() -> dict[str, object]:
    defaults: dict[str, object] = {
        "tickers": DEFAULT_TICKER_TEXT,
        "benchmark": DEFAULT_BENCHMARK,
        "strategy": OPTIMIZATION_STRATEGIES[0],
        "risk_free_rate": DEFAULT_RISK_FREE_RATE,
        "max_weight": DEFAULT_MAX_WEIGHT,
        "rebalance_frequency": REBALANCE_MONTHLY,
        "static_weights": None,
        "prefilled": False,
    }
    result = st.session_state.get("portfolio_result")
    if not isinstance(result, PortfolioAnalysisResult):
        return defaults

    strategy = result.selected_strategy
    if strategy not in OPTIMIZATION_STRATEGIES and strategy != STRATEGY_CUSTOM_MANUAL:
        strategy = OPTIMIZATION_STRATEGIES[0]

    defaults.update(
        {
            "tickers": ", ".join(result.tickers),
            "benchmark": result.benchmark_label or result.benchmark_ticker,
            "strategy": strategy,
            "risk_free_rate": float(getattr(result, "risk_free_rate", DEFAULT_RISK_FREE_RATE)),
            "max_weight": float(getattr(result, "max_weight", DEFAULT_MAX_WEIGHT)),
            "rebalance_frequency": REBALANCE_NONE,
            "static_weights": result.selected_result.weights.copy(),
            "prefilled": True,
        }
    )
    return defaults


def frame_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    data = frame.reset_index()
    data = data.rename(columns={data.columns[0]: "Date"})
    return data.melt(id_vars="Date", var_name="Series", value_name=value_name)


def plot_line_frame(frame: pd.DataFrame, title: str, value_name: str, y_title: str, percent_axis: bool = True) -> alt.Chart:
    chart_data = frame_to_long(frame, value_name)
    axis = alt.Axis(format="%") if percent_axis else alt.Axis(format=",.0f")
    return (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(f"{value_name}:Q", axis=axis, title=y_title),
            color=alt.Color("Series:N"),
            tooltip=[
                alt.Tooltip("Date:T"),
                alt.Tooltip("Series:N"),
                alt.Tooltip(f"{value_name}:Q", format=".2%" if percent_axis else ",.0f"),
            ],
        )
        .properties(title=title, height=360)
    )


def plot_weight_pie(weights: pd.Series) -> alt.Chart:
    chart_data = weights.sort_values(ascending=False).rename("Weight").reset_index()
    chart_data = chart_data.rename(columns={chart_data.columns[0]: "Ticker"})
    chart_data["Label"] = chart_data["Ticker"] + "  " + chart_data["Weight"].map(lambda value: f"{value:.1%}")
    base = alt.Chart(chart_data).encode(
        theta=alt.Theta("Weight:Q", stack=True),
        color=alt.Color("Ticker:N", legend=alt.Legend(title="Ticker")),
    )
    arcs = base.mark_arc(innerRadius=0, outerRadius=145, stroke="white", strokeWidth=1).encode(
        tooltip=[alt.Tooltip("Ticker:N"), alt.Tooltip("Weight:Q", format=".2%")]
    )
    labels = base.mark_text(radius=175, size=12).encode(text="Label:N")
    return (arcs + labels).properties(title="Optimized Portfolio Allocation", height=360)


def plot_risk_contribution(risk_contribution: pd.Series) -> alt.Chart:
    chart_data = risk_contribution.sort_values(ascending=False).rename("Risk Contribution").reset_index()
    chart_data = chart_data.rename(columns={chart_data.columns[0]: "Ticker"})
    return (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Risk Contribution:Q", axis=alt.Axis(format="%"), title="Risk Contribution"),
            y=alt.Y("Ticker:N", sort="-x", title=None),
            color=alt.Color("Ticker:N", legend=None),
            tooltip=[alt.Tooltip("Ticker:N"), alt.Tooltip("Risk Contribution:Q", format=".2%")],
        )
        .properties(title="Risk Contribution", height=360)
    )


def plot_correlation_heatmap(correlation: pd.DataFrame) -> alt.Chart:
    chart_data = correlation.reset_index().rename(columns={correlation.index.name or "index": "Ticker"})
    chart_data = chart_data.melt(id_vars="Ticker", var_name="Peer", value_name="Correlation")
    ordered_tickers = list(correlation.columns)
    return (
        alt.Chart(chart_data)
        .mark_rect()
        .encode(
            x=alt.X("Peer:N", sort=ordered_tickers, title=None),
            y=alt.Y("Ticker:N", sort=ordered_tickers, title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("Ticker:N"),
                alt.Tooltip("Peer:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(title="Ticker Correlation Matrix", height=360)
    )


def plot_drawdown_periods(cumulative_returns: pd.DataFrame, periods: list[tuple[pd.Timestamp, pd.Timestamp, float]]) -> alt.Chart:
    line_data = cumulative_returns[["Portfolio"]].reset_index().rename(columns={cumulative_returns.index.name or "index": "Date"})
    if "Date" not in line_data.columns:
        line_data = line_data.rename(columns={line_data.columns[0]: "Date"})
    period_data = pd.DataFrame(periods, columns=["Start", "End", "Max Drawdown"])
    base = alt.Chart(line_data).encode(x=alt.X("Date:T", title=None))
    shaded = (
        alt.Chart(period_data)
        .mark_rect(color="#D55E00", opacity=0.16)
        .encode(x="Start:T", x2="End:T", tooltip=[alt.Tooltip("Max Drawdown:Q", format=".2%")])
    )
    line = base.mark_line(color="#0072B2", strokeWidth=2).encode(
        y=alt.Y("Portfolio:Q", axis=alt.Axis(format="%"), title="Cumulative Returns"),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Portfolio:Q", format=".2%")],
    )
    zero = base.mark_rule(color="black", strokeDash=[5, 4]).encode(y=alt.datum(0))
    chart = line + zero if period_data.empty else shaded + line + zero
    return chart.properties(title=f"Worst Drawdown Periods | {date_range_label(cumulative_returns.index)}", height=360)


def plot_annual_returns(annual_returns: pd.DataFrame) -> alt.Chart:
    annual_data = annual_returns.reset_index().rename(columns={annual_returns.index.name or "index": "Year"})
    chart_data = annual_data.melt(id_vars="Year", var_name="Series", value_name="Return")
    return (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Year:N", title=None),
            xOffset="Series:N",
            y=alt.Y("Return:Q", axis=alt.Axis(format="%"), title="Annual Return"),
            color=alt.Color("Series:N"),
            tooltip=[alt.Tooltip("Year:N"), alt.Tooltip("Series:N"), alt.Tooltip("Return:Q", format=".2%")],
        )
        .properties(title="Calendar-Year Returns", height=360)
    )


def plot_weights_history(weights_history: pd.DataFrame) -> alt.Chart:
    chart_data = frame_to_long(weights_history, "Weight")
    return (
        alt.Chart(chart_data)
        .mark_area(opacity=0.75)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Weight:Q", stack="normalize", axis=alt.Axis(format="%"), title="Weight"),
            color=alt.Color("Series:N", title="Ticker"),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N", title="Ticker"), alt.Tooltip("Weight:Q", format=".2%")],
        )
        .properties(title="Portfolio Weights Over Time", height=360)
    )


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    percent_keywords = ("Return", "CAGR", "Volatility", "Drawdown", "VaR", "CVaR", "Alpha", "Tracking Error", "Cost", "Turnover")
    ratio_keywords = ("Sharpe", "Sortino", "Beta", "Information Ratio")
    for column in display.columns:
        if column == "Series":
            continue
        if any(keyword in column for keyword in percent_keywords):
            display[column] = display[column].map(lambda value: pct(value) if pd.notna(value) else "N/A")
        elif any(keyword in column for keyword in ratio_keywords):
            display[column] = display[column].map(lambda value: ratio(value) if pd.notna(value) else "N/A")
    return display


def display_portfolio_optimization(result: PortfolioAnalysisResult) -> None:
    selected = result.selected_result
    alpha = result.alpha_metrics
    st.caption(f"Resolved for Yahoo Finance: {result.resolution}")
    st.caption(f"Benchmark resolved for Yahoo Finance: {result.benchmark_label} -> {result.benchmark_ticker}")

    gate_status = "Passed" if alpha.passed else "Rejected"
    st.subheader(f"Alpha Gate: {gate_status}")
    cols = st.columns(4)
    cols[0].metric("Alpha", pct(alpha.alpha_annualized))
    cols[1].metric("Beta", ratio(alpha.beta))
    cols[2].metric("Information ratio", ratio(alpha.information_ratio))
    cols[3].metric("Tracking error", pct(alpha.tracking_error))

    period_cols = st.columns(3)
    period_cols[0].metric("Portfolio period return", pct(result.period_returns["Portfolio"]))
    period_cols[1].metric("Benchmark period return", pct(result.period_returns["Benchmark"]))
    period_cols[2].metric("Overlapping trading days", f"{alpha.observations:,}")

    if alpha.passed:
        st.success("The selected portfolio cleared the configured alpha and information-ratio gate.")
    else:
        st.warning("The selected portfolio did not clear the configured alpha and information-ratio gate.")

    st.subheader(f"Optimized Portfolio: {result.selected_strategy}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Expected return", pct(selected.stats.expected_return))
    metric_cols[1].metric("Volatility", pct(selected.stats.volatility))
    metric_cols[2].metric("Sharpe ratio", ratio(selected.stats.sharpe_ratio))
    metric_cols[3].metric("Optimizer", selected.stats.optimizer)

    weights_df = selected.weights.rename("Weight").reset_index().rename(columns={"index": "Ticker"}).sort_values("Weight", ascending=False)
    weights_df["Weight"] = weights_df["Weight"].map(pct)
    st.dataframe(weights_df, hide_index=True, use_container_width=True)

    allocation_col, drawdown_col = st.columns([0.85, 1.15])
    with allocation_col:
        st.altair_chart(plot_weight_pie(selected.weights), use_container_width=True)
    with drawdown_col:
        st.altair_chart(plot_drawdown_periods(result.cumulative_returns, result.drawdown_periods), use_container_width=True)

    returns_col, cumulative_col = st.columns(2)
    with returns_col:
        st.altair_chart(plot_annual_returns(result.annual_returns), use_container_width=True)
    with cumulative_col:
        st.altair_chart(plot_line_frame(result.cumulative_returns, "Cumulative Returns vs Benchmark", "Return", "Cumulative Returns"), use_container_width=True)

    correlation_col, risk_col = st.columns(2)
    with correlation_col:
        correlation = get_portfolio_correlation_matrix(result)
        st.altair_chart(plot_correlation_heatmap(correlation), use_container_width=True)
        st.caption("High correlation reduces diversification benefits and can make optimizer weights less stable.")
        st.dataframe(correlation.round(2), use_container_width=True)
    with risk_col:
        st.altair_chart(plot_risk_contribution(selected.risk_contribution), use_container_width=True)


def display_strategy_comparison(result: PortfolioAnalysisResult) -> None:
    st.subheader("Strategy Comparison")
    st.dataframe(format_table(result.summary_table), hide_index=True, use_container_width=True)
    cumulative_col, annual_col = st.columns(2)
    with cumulative_col:
        st.altair_chart(plot_line_frame(result.strategy_cumulative_returns, "Strategy Cumulative Returns", "Return", "Cumulative Returns"), use_container_width=True)
    with annual_col:
        st.altair_chart(plot_annual_returns(result.strategy_annual_returns), use_container_width=True)

    weights_table = result.weights_comparison.reset_index()
    risk_table = result.risk_contribution_comparison.reset_index()
    for column in result.weights_comparison.columns:
        weights_table[column] = weights_table[column].map(pct)
        risk_table[column] = risk_table[column].map(pct)

    table_col, risk_table_col = st.columns(2)
    with table_col:
        st.caption("Weights by strategy")
        st.dataframe(weights_table, hide_index=True, use_container_width=True)
    with risk_table_col:
        st.caption("Risk contribution by strategy")
        st.dataframe(risk_table, hide_index=True, use_container_width=True)


def display_described_metric(label: str, value: str, description: str) -> None:
    st.metric(label, value, help=description)


def display_metric_group(
    title: str,
    metrics: list[tuple[str, str, str]],
    columns_count: int,
) -> None:
    st.markdown(f"**{title}**")
    cols = st.columns(columns_count)
    for index, (label, value, metric_description) in enumerate(metrics):
        with cols[index % columns_count]:
            display_described_metric(label, value, metric_description)


def display_monte_carlo(result: PortfolioAnalysisResult, simulations_count: int, horizon_days: int, initial_capital: float, confidence_level: float, seed: int) -> None:
    if not result.alpha_metrics.passed:
        st.warning("Monte Carlo simulation is skipped because the selected portfolio failed the alpha gate.")
        return

    with st.spinner("Running Monte Carlo simulation..."):
        simulations = monte_carlo_simulation(
            result.returns,
            result.selected_result.weights,
            simulations=simulations_count,
            horizon_days=horizon_days,
            initial_portfolio=initial_capital,
            seed=seed,
        )
        risk = calculate_risk_metrics(simulations, initial_portfolio=initial_capital, confidence_level=confidence_level)
        display_values = monte_carlo_display_values(simulations, initial_capital, risk)

    st.subheader("Monte Carlo Simulation")
    st.line_chart(simulations.iloc[:, : min(300, simulations.shape[1])], height=320)
    return_ev = display_values["return_gain_loss_ev"]
    pnl_ev = display_values["pnl_gain_loss_ev"]
    confidence_label = f"{int(confidence_level * 100)}%"
    tail_probability_label = f"{1 - confidence_level:.0%}"
    mean_return = float(display_values["mean_return"])
    std_return = float(display_values["std_return"])
    simulated_sharpe = mean_return / std_return if np.isfinite(mean_return) and np.isfinite(std_return) and std_return != 0 else np.nan

    display_metric_group(
        f"Central forecast ({horizon_days} trading days)",
        [
            (
                "Expected final value",
                money(display_values["expected_final_value"]),
                "Average ending portfolio value across all simulation paths.",
            ),
            (
                "Expected PnL",
                f"{money(pnl_ev.expected_value_gain_loss)} ({pct(return_ev.expected_value_gain_loss)})",
                "Average profit or loss versus starting capital, shown as money and return.",
            ),
            (
                "Mean return",
                pct(mean_return),
                "Average return over the selected horizon across all paths.",
            ),
            (
                "Return std. dev.",
                pct(std_return),
                "Spread of simulated returns. Higher means outcomes are less clustered.",
            ),
            (
                "Simulated Sharpe ratio",
                ratio(simulated_sharpe),
                "Mean simulated return divided by simulated return standard deviation for this horizon.",
            ),
        ],
        columns_count=5,
    )

    display_metric_group(
        "Outcome probabilities",
        [
            (
                "Probability of gain",
                pct(pnl_ev.probability_of_gain),
                "Share of paths that finish above starting capital.",
            ),
            (
                "Probability of loss",
                pct(display_values["probability_of_loss"]),
                "Share of paths that finish below starting capital.",
            ),
            (
                "Probability breakeven",
                pct(pnl_ev.probability_of_breakeven),
                "Share of paths that finish effectively unchanged.",
            ),
            (
                "Probability >= mean return",
                pct(display_values["probability_at_or_above_mean_return"]),
                "Share of paths at or above the average return; this is not the gain probability.",
            ),
        ],
        columns_count=4,
    )

    display_metric_group(
        "Conditional payoff",
        [
            (
                "Average gain if gain",
                f"{money(pnl_ev.average_gain)} ({pct(return_ev.average_gain)})",
                "Mean profit among winning paths only.",
            ),
            (
                "Average loss if loss",
                f"{money(display_values['average_loss_value'])} ({pct(display_values['average_loss_pct'])})",
                "Mean loss among losing paths only, shown as a positive amount.",
            ),
            (
                "Portfolio expected value",
                money(display_values["expected_final_value"]),
                "Starting capital plus expected PnL; equal to the average final portfolio value across paths.",
            ),
        ],
        columns_count=3,
    )

    display_metric_group(
        "Downside tail risk",
        [
            (
                f"VaR {confidence_label}",
                money(display_values["var_value"]),
                f"Loss threshold for this confidence level; roughly {tail_probability_label} of paths lose more.",
            ),
            (
                f"CVaR {confidence_label}",
                money(display_values["cvar_value"]),
                f"Average loss inside the worst {tail_probability_label} of final outcomes.",
            ),
            (
                "Max loss",
                f"{money(display_values['max_loss_value'])} ({pct(display_values['max_loss_pct'])})",
                "Worst final loss seen across all paths; not an intra-period drawdown.",
            ),
        ],
        columns_count=3,
    )
    if abs(pnl_ev.difference_between_methods) > max(1e-8, abs(pnl_ev.expected_value_mean) * 1e-10):
        st.caption(
            "EV method difference: "
            f"{money(pnl_ev.difference_between_methods)} ({pct(return_ev.difference_between_methods)}). "
            "Small differences can come from floating-point precision."
        )

    with st.expander("Expected value formula", expanded=False):
        st.markdown(
            "Expected value is the mean of all Monte Carlo outcomes: `EV = (1 / N) * sum(X_i)`. "
            "The gain/loss decomposition is `EV = P(Gain) * Avg Gain - P(Loss) * Avg Loss`. "
            "Average gain uses only profitable simulations, average loss is the positive magnitude of only losing simulations, "
            "and breakeven paths are tracked separately with zero contribution to EV."
        )


def display_backtest(result) -> None:
    st.caption(f"Resolved for Yahoo Finance: {result.resolution}")
    st.caption(f"Benchmark: {result.benchmark_ticker}")

    strategy_metrics = result.metrics_table[result.metrics_table["Series"] == result.strategy].iloc[0]
    cols = st.columns(5)
    cols[0].metric("Total return", pct(strategy_metrics.get("Total Return", np.nan)))
    cols[1].metric("CAGR", pct(strategy_metrics.get("CAGR", np.nan)))
    cols[2].metric("Sharpe", ratio(strategy_metrics.get("Sharpe Ratio", np.nan)))
    cols[3].metric("Max drawdown", pct(strategy_metrics.get("Max Drawdown", np.nan)))
    cols[4].metric("Final value", money(result.equity_curve[result.strategy].iloc[-1]))

    if result.fallback_count:
        total_rebalances = len(result.rebalance_table)
        st.warning(f"Fallback triggered in {result.fallback_count} out of {total_rebalances} rebalances")
        st.caption("Fallback dates: " + ", ".join(result.fallback_dates))

    st.subheader("Strategy vs Benchmark")
    st.dataframe(format_table(result.comparison_table), hide_index=True, use_container_width=True)

    equity_col, drawdown_col = st.columns(2)
    with equity_col:
        st.altair_chart(plot_line_frame(result.equity_curve, "Equity Curve", "Value", "Portfolio Value", percent_axis=False), use_container_width=True)
    with drawdown_col:
        st.altair_chart(plot_line_frame(result.drawdown, "Drawdown", "Drawdown", "Drawdown"), use_container_width=True)

    rolling_col, weights_col = st.columns(2)
    with rolling_col:
        st.altair_chart(plot_line_frame(result.rolling_sharpe, "Rolling 63-Day Sharpe", "Sharpe", "Sharpe", percent_axis=False), use_container_width=True)
    with weights_col:
        st.altair_chart(plot_weights_history(result.weights_history), use_container_width=True)

    st.subheader("Rebalance History")
    st.dataframe(result.rebalance_table, hide_index=True, use_container_width=True)

    cost_display = result.cost_table.copy()
    for column in ["Turnover", "Transaction Cost", "Slippage Cost", "Total Cost"]:
        if column in cost_display.columns:
            cost_display[column] = cost_display[column].map(pct)
    st.subheader("Turnover and Transaction Costs")
    st.dataframe(cost_display, hide_index=True, use_container_width=True)

    st.subheader("Alpha/Beta Regression")
    st.dataframe(format_table(result.regression_table), hide_index=True, use_container_width=True)


def display_debug(debug_info: dict[str, object]) -> None:
    st.subheader("Debug Panel")
    for section in ["inputs", "data_quality", "return_checks", "metric_checks", "warnings"]:
        with st.expander(section.replace("_", " ").title(), expanded=section == "warnings"):
            st.json(debug_info.get(section, {}))
    with st.expander("Rebalance Logs", expanded=False):
        st.json(debug_info.get("rebalance_logs", []))
    with st.expander("Cost Logs", expanded=False):
        st.json(debug_info.get("cost_logs", []))


st.title("Portfolio Optimization & Backtesting")

portfolio_tab, comparison_tab, backtesting_tab = st.tabs(
    ["Portfolio Optimization", "Strategy Comparison", "Backtesting"]
)

today = date.today()
earliest_date = date(1990, 1, 1)

with portfolio_tab:
    st.subheader("Portfolio Optimization")
    st.caption(TAB_DESCRIPTIONS["Portfolio Optimization"])
    with st.form("portfolio_form"):
        st.markdown("**Data Inputs**")
        ticker_text = st.text_area(
            "Stock names or Yahoo tickers",
            value=DEFAULT_TICKER_TEXT,
            height=92,
            help=UI_HELP["tickers"],
        )
        input_cols = st.columns(3)
        with input_cols[0]:
            start_date = st.date_input(
                "Start date",
                value=today - timedelta(days=365 * 3),
                min_value=earliest_date,
                max_value=today,
                help=UI_HELP["start_date"],
            )
        with input_cols[1]:
            end_date = st.date_input(
                "End date",
                value=today,
                min_value=earliest_date,
                max_value=today,
                help=UI_HELP["end_date"],
            )
        with input_cols[2]:
            benchmark = st.text_input(
                "Benchmark name or ticker",
                value=DEFAULT_BENCHMARK,
                help=UI_HELP["benchmark"],
            )

        st.markdown("**Optimization Settings**")
        custom_cols = st.columns([0.34, 0.33, 0.33])
        with custom_cols[0]:
            enable_custom_weights = st.checkbox(
                "Enable Custom Manual Weights",
                value=False,
                help=UI_HELP["enable_custom_weights"],
            )
        with custom_cols[1]:
            normalize_custom_weights = st.checkbox(
                "Normalize custom weights",
                value=True,
                help=UI_HELP["normalize_weights"],
            )
        custom_weights_df = pd.DataFrame(columns=["Ticker", "Custom Weight %"])
        if enable_custom_weights:
            try:
                preview_tickers = normalize_tickers(ticker_text)
                st.caption("Custom weights are percentages. Example: 25 means 25% of the portfolio.")
                custom_weights_df = st.data_editor(
                    manual_weights_template(preview_tickers),
                    key=f"portfolio_custom_weights_{abs(hash(tuple(preview_tickers))) % 1000000}",
                    hide_index=True,
                    use_container_width=True,
                    disabled=["Ticker"],
                    column_config={
                        "Ticker": st.column_config.TextColumn(
                            "Ticker",
                            help="Resolved Yahoo ticker. Unit: ticker symbol.",
                        ),
                        "Custom Weight %": st.column_config.NumberColumn(
                            "Custom Weight %",
                            help="Manual allocation for each ticker. Unit: percent of portfolio. Default: equal split.",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                        ),
                    },
                )
            except PortfolioError as exc:
                st.warning(f"Custom weight editor is waiting for valid portfolio tickers: {exc}")

        selectable_strategies = list(OPTIMIZATION_STRATEGIES)
        if enable_custom_weights:
            selectable_strategies.append(STRATEGY_CUSTOM_MANUAL)

        opt_cols = st.columns(4)
        with opt_cols[0]:
            selected_strategy = st.selectbox(
                "Optimization strategy",
                options=selectable_strategies,
                help=UI_HELP["strategy"],
            )
        with opt_cols[1]:
            risk_free_rate = st.number_input(
                "Risk-free rate (annual decimal)",
                min_value=0.0,
                max_value=0.25,
                value=DEFAULT_RISK_FREE_RATE,
                step=0.005,
                help=UI_HELP["risk_free_rate"],
            )
        with opt_cols[2]:
            max_weight = st.slider(
                "Max single-stock weight (fraction)",
                min_value=0.05,
                max_value=1.0,
                value=DEFAULT_MAX_WEIGHT,
                step=0.05,
                help=UI_HELP["max_weight"],
            )

        st.markdown("**Strategy Filters**")
        val_cols = st.columns(4)
        with val_cols[0]:
            alpha_threshold = st.number_input(
                "Alpha threshold (annual decimal)",
                value=0.0,
                step=0.01,
                help=UI_HELP["alpha_threshold"],
            )
        with val_cols[1]:
            information_ratio_threshold = st.number_input(
                "Information ratio threshold (ratio)",
                value=0.5,
                step=0.1,
                help=UI_HELP["information_ratio_threshold"],
            )

        st.markdown("**Simulation Settings**")
        sim_cols = st.columns(4)
        with sim_cols[0]:
            mc_sims = st.number_input(
                "Simulations (count)",
                min_value=100,
                max_value=100000,
                value=20000,
                step=1000,
                help=UI_HELP["simulations"],
            )
        with sim_cols[1]:
            horizon_days = st.number_input(
                "Time horizon (trading days)",
                min_value=1,
                max_value=756,
                value=60,
                step=5,
                help=UI_HELP["time_horizon"],
            )
        with sim_cols[2]:
            confidence_level = st.select_slider(
                "Confidence level (probability)",
                options=[0.90, 0.95, 0.99],
                value=0.95,
                help=UI_HELP["confidence_level"],
            )
        with sim_cols[3]:
            seed = st.number_input(
                "Random seed (integer)",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                help=UI_HELP["random_seed"],
            )
        if mc_sims > 50000:
            st.warning("High simulation count selected. Monte Carlo may run slowly above 50,000 paths.")

        st.markdown("**Capital**")
        initial_portfolio = st.number_input(
            "Initial portfolio (currency)",
            min_value=1000,
            value=int(DEFAULT_INITIAL_CAPITAL),
            step=5000,
            help=UI_HELP["initial_portfolio"],
        )

        with st.expander("Advanced Settings", expanded=False):
            l2_reg = st.number_input(
                "L2 regularization (penalty coefficient)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.05,
                help=UI_HELP["l2_reg"],
            )

        run_portfolio = st.form_submit_button(
            "Run portfolio pipeline",
            type="primary",
            use_container_width=True,
            help="Run optimization, validation metrics, strategy comparisons, and Monte Carlo output.",
        )

    if run_portfolio:
        progress = st.progress(0)
        status = st.status("Starting portfolio pipeline...", expanded=True)
        try:
            status.write("1. Validating inputs")
            progress.progress(10)
            status.write("2. Loading historical price data")
            status.write("3. Cleaning returns and optimizing strategies")
            result = run_portfolio_analysis(
                raw_tickers=ticker_text,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                selected_strategy=selected_strategy,
                benchmark=benchmark,
                risk_free_rate=float(risk_free_rate),
                max_weight=float(max_weight),
                l2_reg=float(l2_reg),
                enable_custom_weights=enable_custom_weights,
                custom_weights_df=custom_weights_df,
                normalize_custom_weights=normalize_custom_weights,
                previous_weights=st.session_state.get("latest_strategy_weights", {}),
                alpha_threshold=float(alpha_threshold),
                information_ratio_threshold=float(information_ratio_threshold),
            )
            progress.progress(90)
            st.session_state["portfolio_result"] = result
            st.session_state["latest_strategy_weights"] = {
                strategy: strategy_result.weights for strategy, strategy_result in result.strategy_results.items()
            }
            st.session_state["portfolio_mc_inputs"] = {
                "simulations_count": int(mc_sims),
                "horizon_days": int(horizon_days),
                "initial_capital": float(initial_portfolio),
                "confidence_level": float(confidence_level),
                "seed": int(seed),
            }
            progress.progress(100)
            status.update(label="Portfolio pipeline completed.", state="complete", expanded=False)
        except PortfolioError as exc:
            st.session_state.pop("portfolio_result", None)
            status.update(label="Portfolio pipeline stopped.", state="error", expanded=True)
            st.error(str(exc))
        except Exception:
            status.update(label="Portfolio pipeline failed.", state="error", expanded=True)
            raise

    if "portfolio_result" in st.session_state:
        display_portfolio_optimization(st.session_state["portfolio_result"])
        display_monte_carlo(st.session_state["portfolio_result"], **st.session_state["portfolio_mc_inputs"])
    else:
        st.info("Set the inputs, then run the portfolio pipeline.")

with comparison_tab:
    st.caption(TAB_DESCRIPTIONS["Strategy Comparison"])
    if "portfolio_result" in st.session_state:
        display_strategy_comparison(st.session_state["portfolio_result"])
    else:
        st.info("Run the portfolio pipeline first to compare strategies.")

with backtesting_tab:
    st.subheader("Portfolio Backtesting")
    st.caption(TAB_DESCRIPTIONS["Backtesting"])
    st.caption(
        "Use None to hold supplied weights through the full test period. "
        "Weekly, Monthly, and Quarterly use walk-forward training, where the first realized period starts only after the selected lookback window."
    )
    bt_defaults = backtest_prefill_defaults()
    if bt_defaults["prefilled"]:
        st.caption(
            "Defaults are prefilled from the latest optimizer result; None will hold those optimized weights until the end of the backtest."
        )
    with st.form("backtesting_form"):
        st.markdown("**Data Inputs**")
        bt_tickers = st.text_area(
            "Backtest tickers",
            value=str(bt_defaults["tickers"]),
            height=92,
            help=UI_HELP["tickers"],
        )
        bt_cols = st.columns(3)
        with bt_cols[0]:
            bt_start = st.date_input(
                "Backtest start date",
                value=today - timedelta(days=365 * 6),
                min_value=earliest_date,
                max_value=today,
                help=UI_HELP["start_date"],
            )
        with bt_cols[1]:
            bt_end = st.date_input(
                "Backtest end date",
                value=today,
                min_value=earliest_date,
                max_value=today,
                help=UI_HELP["end_date"],
            )
        with bt_cols[2]:
            bt_benchmark = st.text_input(
                "Benchmark ticker",
                value=str(bt_defaults["benchmark"]),
                help=UI_HELP["benchmark"],
            )

        st.markdown("**Optimization Settings**")
        bt_custom_cols = st.columns([0.25, 0.25, 0.25])
        with bt_custom_cols[0]:
            bt_enable_custom = st.checkbox(
                "Enable custom manual weights",
                value=False,
                help=UI_HELP["enable_custom_weights"],
            )
        with bt_custom_cols[1]:
            bt_normalize_custom = st.checkbox(
                "Normalize backtest weights",
                value=True,
                help=UI_HELP["normalize_weights"],
            )
        with bt_custom_cols[2]:
            bt_comparison_mode = st.checkbox(
                "Enable comparison mode",
                value=True,
                help=UI_HELP["comparison_mode"],
            )

        bt_custom_df = pd.DataFrame(columns=["Ticker", "Custom Weight %"])
        if bt_enable_custom:
            try:
                bt_preview_tickers = normalize_tickers(bt_tickers)
                st.caption("Backtest custom weights are percentages.")
                bt_custom_df = st.data_editor(
                    manual_weights_template(bt_preview_tickers),
                    key=f"backtest_custom_weights_{abs(hash(tuple(bt_preview_tickers))) % 1000000}",
                    hide_index=True,
                    use_container_width=True,
                    disabled=["Ticker"],
                    column_config={
                        "Ticker": st.column_config.TextColumn(
                            "Ticker",
                            help="Resolved Yahoo ticker. Unit: ticker symbol.",
                        ),
                        "Custom Weight %": st.column_config.NumberColumn(
                            "Custom Weight %",
                            help="Manual allocation for the backtest. Unit: percent of portfolio.",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                        ),
                    },
                )
                if bt_defaults["strategy"] == STRATEGY_CUSTOM_MANUAL:
                    st.caption("Custom manual backtests use the weights entered here when custom weights are enabled.")
            except PortfolioError as exc:
                st.warning(f"Custom weight editor is waiting for valid backtest tickers: {exc}")

        bt_strategy_options = list(OPTIMIZATION_STRATEGIES)
        if bt_enable_custom or bt_defaults["strategy"] == STRATEGY_CUSTOM_MANUAL:
            bt_strategy_options.append(STRATEGY_CUSTOM_MANUAL)

        setting_cols = st.columns(4)
        with setting_cols[0]:
            bt_strategy_default = str(bt_defaults["strategy"])
            if bt_strategy_default not in bt_strategy_options:
                bt_strategy_default = OPTIMIZATION_STRATEGIES[0]
            bt_strategy = st.selectbox(
                "Backtest strategy",
                options=bt_strategy_options,
                index=bt_strategy_options.index(bt_strategy_default),
                help=UI_HELP["strategy"],
            )
        with setting_cols[1]:
            bt_rebalance_options = list(SUPPORTED_REBALANCE_FREQUENCIES)
            bt_rebalance_default = str(bt_defaults["rebalance_frequency"])
            if bt_rebalance_default not in bt_rebalance_options:
                bt_rebalance_default = REBALANCE_MONTHLY
            bt_rebalance = st.selectbox(
                "Rebalance frequency",
                options=bt_rebalance_options,
                index=bt_rebalance_options.index(bt_rebalance_default),
                help=UI_HELP["rebalance_frequency"],
            )
        with setting_cols[2]:
            bt_lookback_options = list(LOOKBACK_LABELS)
            bt_lookback = st.selectbox(
                "Lookback window",
                options=bt_lookback_options,
                index=bt_lookback_options.index(DEFAULT_LOOKBACK_DAYS),
                format_func=lambda value: LOOKBACK_LABELS[int(value)],
                help=UI_HELP["lookback_window"],
            )
        with setting_cols[3]:
            bt_max_weight = st.slider(
                "Backtest max weight (fraction)",
                min_value=0.05,
                max_value=1.0,
                value=float(bt_defaults["max_weight"]),
                step=0.05,
                help=UI_HELP["max_weight"],
            )

        bt_risk_free = st.number_input(
            "Backtest risk-free rate (annual decimal)",
            min_value=0.0,
            max_value=0.25,
            value=float(bt_defaults["risk_free_rate"]),
            step=0.005,
            help=UI_HELP["risk_free_rate"],
        )

        bt_static_weights = bt_defaults.get("static_weights")
        if bt_rebalance == REBALANCE_NONE:
            if not bt_enable_custom and isinstance(bt_static_weights, pd.Series):
                st.markdown("**Static Optimized Allocation**")
                st.dataframe(weights_display_frame(bt_static_weights), hide_index=True, use_container_width=True)
            elif not bt_enable_custom:
                st.warning("Run the portfolio pipeline first or enable custom manual weights to use None.")
            if bt_comparison_mode:
                st.caption("With None, comparison mode uses only Benchmark and Equal Weight Basket.")

        required_price_rows = 3 if bt_rebalance == REBALANCE_NONE else int(bt_lookback) + 2
        available_business_days = estimated_business_days(bt_start, bt_end)
        if available_business_days < required_price_rows:
            if bt_rebalance == REBALANCE_NONE:
                st.warning(
                    f"The selected dates contain roughly {available_business_days} business days, "
                    f"but None needs at least {required_price_rows} price rows."
                )
            else:
                st.warning(
                    f"The selected dates contain roughly {available_business_days} business days, "
                    f"but {LOOKBACK_LABELS[int(bt_lookback)]} needs at least {required_price_rows} price rows. "
                    "Move the start date earlier or choose a shorter lookback."
                )
        if bt_rebalance == REBALANCE_NONE:
            st.caption("None holds supplied weights over the full selected period and ignores the lookback window.")
        else:
            st.caption(
                f"Selected lookback requires at least {required_price_rows} usable price rows for every ticker "
                "before the first out-of-sample period can be created."
            )

        st.markdown("**Capital**")
        bt_capital = st.number_input(
            "Initial capital (currency)",
            min_value=1000,
            value=int(DEFAULT_INITIAL_CAPITAL),
            step=5000,
            key="bt_capital",
            help=UI_HELP["initial_portfolio"],
        )

        with st.expander("Advanced Settings", expanded=False):
            cost_cols = st.columns(3)
            with cost_cols[0]:
                bt_transaction_cost_pct = st.number_input(
                    "Transaction cost (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=DEFAULT_TRANSACTION_COST * 100,
                    step=0.05,
                    help=UI_HELP["transaction_cost"],
                )
            with cost_cols[1]:
                bt_slippage_bps = st.number_input(
                    "Slippage (bps)",
                    min_value=0.0,
                    max_value=500.0,
                    value=DEFAULT_SLIPPAGE * 10000,
                    step=1.0,
                    help=UI_HELP["slippage"],
                )
            with cost_cols[2]:
                bt_debug = st.checkbox(
                    "Enable debug mode",
                    value=False,
                    help=UI_HELP["debug_mode"],
                )

        run_backtest = st.form_submit_button(
            "Run Backtest",
            type="primary",
            use_container_width=True,
            help="Run the selected static-hold or walk-forward backtest and render performance, turnover, fallback, and debug output.",
        )

    if run_backtest:
        progress = st.progress(0)
        status = st.status("Starting backtest...", expanded=True)
        try:
            static_weights_for_backtest = None
            if bt_rebalance == REBALANCE_NONE and not bt_enable_custom and isinstance(bt_static_weights, pd.Series):
                static_weights_for_backtest = bt_static_weights
            run_strategy = STRATEGY_CUSTOM_MANUAL if bt_rebalance == REBALANCE_NONE and bt_enable_custom else bt_strategy
            stages = [
                "1. Validating inputs",
                "2. Loading historical price data",
                "3. Cleaning and aligning data",
                "4. Calculating returns",
                "5. Applying static weights" if bt_rebalance == REBALANCE_NONE else "5. Running walk-forward optimization",
                "6. Applying transaction costs",
                "7. Calculating benchmark returns",
                "8. Calculating metrics",
                "9. Preparing dashboard outputs",
            ]
            for idx, stage in enumerate(stages[:4], start=1):
                status.write(stage)
                progress.progress(idx * 8)
            result = run_walk_forward_backtest(
                tickers=bt_tickers,
                benchmark=bt_benchmark,
                start=bt_start.isoformat(),
                end=bt_end.isoformat(),
                strategy=run_strategy,
                rebalance_frequency=bt_rebalance,
                lookback_days=int(bt_lookback),
                initial_capital=float(bt_capital),
                transaction_cost=float(bt_transaction_cost_pct) / 100.0,
                slippage=float(bt_slippage_bps) / 10000.0,
                risk_free_rate=float(bt_risk_free),
                max_weight=float(bt_max_weight),
                custom_weights_df=bt_custom_df if bt_enable_custom else None,
                normalize_custom_weights=bt_normalize_custom,
                static_weights=static_weights_for_backtest,
                comparison_mode=bt_comparison_mode,
                debug=bt_debug,
            )
            for idx, stage in enumerate(stages[4:], start=5):
                status.write(stage)
                progress.progress(min(idx * 10, 95))
            st.session_state["backtest_result"] = result
            st.session_state["backtest_debug_enabled"] = bt_debug
            progress.progress(100)
            status.update(label="Backtest complete.", state="complete", expanded=False)
        except PortfolioError as exc:
            st.session_state.pop("backtest_result", None)
            status.write("Backtest stopped while validating inputs or data.")
            status.update(label="Backtest stopped.", state="error", expanded=True)
            st.error(str(exc))
            if bt_rebalance == REBALANCE_NONE:
                st.info(
                    "For None, the backtest needs static weights that exactly match the selected tickers. "
                    "Run the portfolio pipeline first, keep the prefilled ticker list, or enable custom manual weights."
                )
            else:
                st.info(
                    f"For {LOOKBACK_LABELS[int(bt_lookback)]}, the loader needs at least {int(bt_lookback) + 2} usable price rows "
                    "for every selected ticker. Move the backtest start date earlier, choose a shorter lookback, "
                    "or remove tickers with shorter Yahoo Finance history."
                )
        except Exception:
            status.update(label="Backtest failed.", state="error", expanded=True)
            raise

    if "backtest_result" in st.session_state:
        display_backtest(st.session_state["backtest_result"])
        if st.session_state.get("backtest_debug_enabled", False):
            display_debug(st.session_state["backtest_result"].debug_info)
    else:
        st.info("Set the backtest inputs, then run the backtest.")
