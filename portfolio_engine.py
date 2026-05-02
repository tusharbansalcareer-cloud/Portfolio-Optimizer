from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from core.constants import (
    DEFAULT_BENCHMARK,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_FREE_RATE,
    OPTIMIZATION_STRATEGIES,
    STRATEGY_CUSTOM_MANUAL,
    STRATEGY_MEAN_VARIANCE,
)
from core.data_utils import (
    align_strategy_and_benchmark_returns,
    calculate_simple_returns,
    fetch_prices,
    get_benchmark_returns as _get_benchmark_returns,
)
from core.metrics import (
    AlphaMetrics,
    PortfolioStats,
    RiskMetrics,
    annual_returns,
    calculate_risk_metrics_from_simulations,
    clean_returns,
    compute_portfolio_returns,
    compute_risk_contribution,
    compute_turnover,
    correlation_matrix as build_correlation_matrix,
    cumulative_returns,
    estimate_annualized_inputs,
    evaluate_alpha,
    max_drawdown,
    risk_metrics_display_values,
    total_return,
    worst_drawdown_periods,
)
from core.optimizer import OptimizationResult, PortfolioOptimizer
from core.validation import (
    PortfolioError,
    describe_ticker_resolution,
    normalize_strategy_name,
    normalize_ticker,
    normalize_tickers,
    validate_custom_weights,
    validate_date_range,
)


@dataclass(frozen=True)
class StrategyResult:
    """Portfolio strategy result used by the optimization dashboard."""

    strategy: str
    weights: pd.Series
    stats: PortfolioStats
    risk_contribution: pd.Series
    portfolio_returns: pd.Series
    cumulative_return: float
    max_drawdown: float
    annual_returns: pd.Series
    turnover: float | None
    optimizer_success: bool = True
    fallback_used: bool = False
    optimizer_message: str = ""


@dataclass(frozen=True)
class PortfolioAnalysisResult:
    """Fully prepared dashboard payload for the optimization and comparison tabs."""

    tickers: list[str]
    benchmark_ticker: str
    benchmark_label: str
    resolution: str
    returns: pd.DataFrame
    correlation_matrix: pd.DataFrame
    benchmark_returns: pd.Series
    risk_free_rate: float
    max_weight: float
    strategy_results: dict[str, StrategyResult]
    selected_strategy: str
    selected_result: StrategyResult
    alpha_metrics: AlphaMetrics
    aligned_returns: pd.DataFrame
    period_returns: dict[str, float]
    cumulative_returns: pd.DataFrame
    annual_returns: pd.DataFrame
    drawdown_periods: list[tuple[pd.Timestamp, pd.Timestamp, float]]
    summary_table: pd.DataFrame
    strategy_returns: pd.DataFrame
    strategy_cumulative_returns: pd.DataFrame
    strategy_annual_returns: pd.DataFrame
    weights_comparison: pd.DataFrame
    risk_contribution_comparison: pd.DataFrame


def run_portfolio_analysis(
    raw_tickers: str | Iterable[str],
    start: str,
    end: str,
    selected_strategy: str,
    benchmark: str = DEFAULT_BENCHMARK,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    l2_reg: float = 0.0,
    enable_custom_weights: bool = False,
    custom_weights_df: pd.DataFrame | None = None,
    normalize_custom_weights: bool = True,
    previous_weights: Mapping[str, pd.Series] | None = None,
    alpha_threshold: float = 0.0,
    information_ratio_threshold: float = 0.5,
) -> PortfolioAnalysisResult:
    """Run the optimization dashboard pipeline and return display-ready outputs."""
    validate_date_range(start, end)
    tickers = normalize_tickers(raw_tickers)
    benchmark_ticker = normalize_ticker(benchmark)
    resolution = describe_ticker_resolution(raw_tickers, tickers)
    prices = get_data(tickers, start, end)
    returns = compute_returns(prices)
    benchmark_returns = get_benchmark_returns(benchmark_ticker, start, end)
    previous_weights = previous_weights or {}

    strategy_results = compare_strategies(
        returns,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        l2_reg=l2_reg,
        previous_weights=previous_weights,
    )

    custom_weights = None
    if enable_custom_weights:
        custom_weights = get_custom_weights(custom_weights_df, tickers, normalize=normalize_custom_weights)
        strategy_results[STRATEGY_CUSTOM_MANUAL] = build_custom_strategy_result(
            returns,
            custom_weights,
            risk_free_rate=risk_free_rate,
            previous_weights=previous_weights.get(STRATEGY_CUSTOM_MANUAL),
        )

    selected_strategy = normalize_strategy_name(selected_strategy)
    if selected_strategy == STRATEGY_CUSTOM_MANUAL and not enable_custom_weights:
        raise PortfolioError("Custom manual allocation is selected but custom weights are disabled.")
    if selected_strategy not in strategy_results:
        raise PortfolioError("Selected strategy is not available. Run the pipeline again with matching inputs.")

    selected_result = strategy_results[selected_strategy]
    alpha_metrics = evaluate_alpha(
        selected_result.portfolio_returns,
        benchmark_returns,
        alpha_threshold=alpha_threshold,
        information_ratio_threshold=information_ratio_threshold,
    )
    aligned_returns = align_strategy_and_benchmark_returns(selected_result.portfolio_returns, benchmark_returns)
    period_returns = {
        "Portfolio": total_return(aligned_returns["Portfolio"]),
        "Benchmark": total_return(aligned_returns["Benchmark"]),
    }
    strategy_returns = strategy_returns_frame(strategy_results)

    return PortfolioAnalysisResult(
        tickers=tickers,
        benchmark_ticker=benchmark_ticker,
        benchmark_label=str(benchmark).strip(),
        resolution=resolution,
        returns=returns,
        correlation_matrix=build_correlation_matrix(returns),
        benchmark_returns=benchmark_returns,
        risk_free_rate=float(risk_free_rate),
        max_weight=float(max_weight),
        strategy_results=strategy_results,
        selected_strategy=selected_strategy,
        selected_result=selected_result,
        alpha_metrics=alpha_metrics,
        aligned_returns=aligned_returns,
        period_returns=period_returns,
        cumulative_returns=cumulative_returns(aligned_returns),
        annual_returns=annual_returns(aligned_returns),
        drawdown_periods=worst_drawdown_periods(aligned_returns["Portfolio"]),
        summary_table=comparison_summary_frame(strategy_results),
        strategy_returns=strategy_returns,
        strategy_cumulative_returns=cumulative_returns(strategy_returns),
        strategy_annual_returns=annual_returns(strategy_returns),
        weights_comparison=strategy_weights_frame(strategy_results),
        risk_contribution_comparison=strategy_risk_contribution_frame(strategy_results),
    )


def get_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Compatibility wrapper around shared price loading."""
    return fetch_prices(tickers, start, end)


def get_benchmark_returns(benchmark: str, start: str, end: str) -> pd.Series:
    """Compatibility wrapper around shared benchmark loading."""
    return _get_benchmark_returns(benchmark, start, end)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around shared simple-return calculation."""
    return calculate_simple_returns(prices, min_assets=2)


def get_portfolio_correlation_matrix(result: PortfolioAnalysisResult) -> pd.DataFrame:
    """Return a result correlation matrix, rebuilding it for stale Streamlit session objects."""
    correlation = getattr(result, "correlation_matrix", None)
    if isinstance(correlation, pd.DataFrame) and not correlation.empty:
        return correlation
    return build_correlation_matrix(result.returns)


def optimize_mean_variance(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    risk_aversion: float = 5.0,
    l2_reg: float = 0.0,
) -> tuple[pd.Series, PortfolioStats]:
    """Compatibility wrapper for the canonical Max Sharpe optimizer."""
    result = PortfolioOptimizer().optimize_max_sharpe(returns, risk_free_rate, max_weight, l2_reg)
    return result.weights, result.stats


def optimize_minimum_variance(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, PortfolioStats]:
    """Compatibility wrapper for minimum variance."""
    result = PortfolioOptimizer().optimize_min_variance(returns, risk_free_rate, max_weight)
    return result.weights, result.stats


def optimize_risk_parity(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, PortfolioStats]:
    """Compatibility wrapper for risk parity."""
    result = PortfolioOptimizer().optimize_risk_parity(returns, risk_free_rate, max_weight)
    return result.weights, result.stats


def optimize_equal_weight(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, PortfolioStats]:
    """Compatibility wrapper for equal weight."""
    result = PortfolioOptimizer().optimize_equal_weight(returns, risk_free_rate, max_weight)
    return result.weights, result.stats


def optimize_strategy(
    strategy: str,
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    risk_aversion: float = 5.0,
    l2_reg: float = 0.0,
    previous_weights: pd.Series | None = None,
) -> StrategyResult:
    """Optimize one strategy and package the shared metrics into a StrategyResult."""
    optimizer_result = PortfolioOptimizer().optimize_strategy_weights(
        strategy,
        returns,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        l2_reg=l2_reg,
        allow_equal_weight_fallback=True,
    )
    return build_strategy_result(
        normalize_strategy_name(strategy),
        returns,
        optimizer_result,
        previous_weights=previous_weights,
    )


def compare_strategies(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    risk_aversion: float = 5.0,
    l2_reg: float = 0.0,
    previous_weights: Mapping[str, pd.Series] | None = None,
) -> dict[str, StrategyResult]:
    """Optimize each supported strategy using the single PortfolioOptimizer implementation."""
    previous_weights = previous_weights or {}
    results = {}
    for strategy in OPTIMIZATION_STRATEGIES:
        results[strategy] = optimize_strategy(
            strategy,
            returns,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            l2_reg=l2_reg,
            previous_weights=previous_weights.get(strategy),
        )
    return results


def get_custom_weights(
    custom_weights_df: pd.DataFrame,
    tickers: list[str],
    normalize: bool = True,
) -> pd.Series:
    """Validate manual portfolio weights and return proportions indexed by ticker."""
    return validate_custom_weights(custom_weights_df, tickers, normalize=normalize)


def build_custom_strategy_result(
    returns: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    previous_weights: pd.Series | None = None,
) -> StrategyResult:
    """Build a StrategyResult for user-provided static manual weights."""
    optimizer_result = PortfolioOptimizer().optimize_strategy_weights(
        STRATEGY_CUSTOM_MANUAL,
        returns,
        risk_free_rate=risk_free_rate,
        custom_weights=weights,
    )
    return build_strategy_result(
        STRATEGY_CUSTOM_MANUAL,
        returns,
        optimizer_result,
        previous_weights=previous_weights,
    )


def build_strategy_result(
    strategy: str,
    returns: pd.DataFrame,
    optimizer_result: OptimizationResult,
    previous_weights: pd.Series | None = None,
) -> StrategyResult:
    """Create one strategy result without duplicating metric calculations."""
    clean = clean_returns(returns)
    _, covariance = estimate_annualized_inputs(clean)
    portfolio_returns = compute_portfolio_returns(clean, optimizer_result.weights)
    return StrategyResult(
        strategy=strategy,
        weights=optimizer_result.weights,
        stats=optimizer_result.stats,
        risk_contribution=compute_risk_contribution(optimizer_result.weights, covariance),
        portfolio_returns=portfolio_returns,
        cumulative_return=total_return(portfolio_returns),
        max_drawdown=max_drawdown(portfolio_returns),
        annual_returns=annual_returns(portfolio_returns),
        turnover=compute_turnover(optimizer_result.weights, previous_weights),
        optimizer_success=optimizer_result.optimizer_success,
        fallback_used=optimizer_result.fallback_used,
        optimizer_message=optimizer_result.message,
    )


def comparison_summary_frame(results: Mapping[str, StrategyResult]) -> pd.DataFrame:
    """Build the strategy comparison table."""
    rows = []
    for strategy, result in results.items():
        rows.append(
            {
                "Strategy": strategy,
                "Expected Annual Return": result.stats.expected_return,
                "Annual Volatility": result.stats.volatility,
                "Sharpe Ratio": result.stats.sharpe_ratio,
                "Max Drawdown": result.max_drawdown,
                "Cumulative Return": result.cumulative_return,
                "Turnover": np.nan if result.turnover is None else result.turnover,
                "Optimizer": result.stats.optimizer,
                "Fallback Used": result.fallback_used,
            }
        )
    return pd.DataFrame(rows)


def strategy_returns_frame(results: Mapping[str, StrategyResult]) -> pd.DataFrame:
    """Return aligned realized returns for each strategy."""
    series = [result.portfolio_returns.rename(strategy) for strategy, result in results.items()]
    return pd.concat(series, axis=1, join="inner").dropna()


def strategy_weights_frame(results: Mapping[str, StrategyResult]) -> pd.DataFrame:
    """Return strategy weights as a ticker-by-strategy table."""
    series = [result.weights.rename(strategy) for strategy, result in results.items()]
    frame = pd.concat(series, axis=1).fillna(0.0)
    frame.index.name = "Ticker"
    return frame


def strategy_risk_contribution_frame(results: Mapping[str, StrategyResult]) -> pd.DataFrame:
    """Return risk contribution as a ticker-by-strategy table."""
    series = [result.risk_contribution.rename(strategy) for strategy, result in results.items()]
    frame = pd.concat(series, axis=1).fillna(0.0)
    frame.index.name = "Ticker"
    return frame


def optimize_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[pd.Series, PortfolioStats]:
    """Compatibility wrapper for the main optimizer."""
    result = optimize_strategy(STRATEGY_MEAN_VARIANCE, returns, risk_free_rate=risk_free_rate, max_weight=max_weight)
    return result.weights, result.stats


def compute_cumulative_return(returns: pd.Series) -> float:
    """Compatibility wrapper for total return."""
    return total_return(returns)


def compute_annual_returns(returns: pd.Series) -> pd.Series:
    """Compatibility wrapper for calendar-year returns."""
    return annual_returns(returns)


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compatibility wrapper for max drawdown."""
    return max_drawdown(returns)


def monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: pd.Series,
    simulations: int = 64000,
    horizon_days: int = 60,
    initial_portfolio: float = DEFAULT_INITIAL_CAPITAL,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Simulate future portfolio values from the empirical mean/covariance of simple returns."""
    if simulations < 100:
        raise PortfolioError("Use at least 100 simulations.")
    if horizon_days < 1:
        raise PortfolioError("Time horizon must be at least one day.")
    if initial_portfolio <= 0:
        raise PortfolioError("Initial portfolio value must be greater than zero.")

    aligned_weights = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    mean_returns = returns.mean().to_numpy()
    covariance = returns.cov().to_numpy()
    rng = np.random.default_rng(seed)

    daily_asset_returns = rng.multivariate_normal(mean_returns, covariance, size=(horizon_days, simulations))
    daily_portfolio_returns = daily_asset_returns @ aligned_weights
    portfolio_paths = initial_portfolio * np.cumprod(1 + daily_portfolio_returns, axis=0)

    index = pd.RangeIndex(1, horizon_days + 1, name="day")
    columns = pd.RangeIndex(1, simulations + 1, name="simulation")
    return pd.DataFrame(portfolio_paths, index=index, columns=columns)


def calculate_risk_metrics(
    simulations: pd.DataFrame,
    initial_portfolio: float,
    confidence_level: float = 0.95,
) -> RiskMetrics:
    """Compatibility wrapper for Monte Carlo risk metrics."""
    return calculate_risk_metrics_from_simulations(simulations, initial_portfolio, confidence_level)


def monte_carlo_display_values(
    simulations: pd.DataFrame,
    initial_portfolio: float,
    risk,
) -> dict[str, float | object]:
    """Compatibility wrapper for Monte Carlo values rendered by Streamlit."""
    return risk_metrics_display_values(simulations, initial_portfolio, risk)
