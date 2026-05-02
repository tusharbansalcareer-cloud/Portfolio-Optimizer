from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.constants import TRADING_DAYS
from core.validation import PortfolioError


@dataclass(frozen=True)
class PortfolioStats:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimizer: str


@dataclass(frozen=True)
class AlphaMetrics:
    alpha_annualized: float
    beta: float
    information_ratio: float
    excess_return_annualized: float
    tracking_error: float
    observations: int
    passed: bool
    alpha_p_value: float | None = None


@dataclass(frozen=True)
class RiskMetrics:
    mean_final_value: float
    expected_final_value: float
    mean_return: float
    std_return: float
    probability_at_or_above_mean_return: float
    probability_of_loss: float
    average_loss_pct: float
    average_loss_value: float
    var_pct: float
    cvar_pct: float
    var_value: float
    cvar_value: float
    max_loss_pct: float
    max_loss_value: float
    return_gain_loss_ev: GainLossExpectedValue
    pnl_gain_loss_ev: GainLossExpectedValue


@dataclass(frozen=True)
class GainLossExpectedValue:
    """Expected value decomposition for a finite vector of simulated outcomes."""

    total_simulations: int
    probability_of_gain: float
    probability_of_loss: float
    probability_of_breakeven: float
    average_gain: float
    average_loss: float
    expected_value_mean: float
    expected_value_gain_loss: float
    difference_between_methods: float
    median_outcome: float
    best_outcome: float
    worst_outcome: float


def clean_returns(returns: pd.DataFrame, min_assets: int = 2) -> pd.DataFrame:
    """Clean return data for optimization and portfolio metrics."""
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.empty:
        raise PortfolioError("Could not compute clean returns for optimization.")
    if clean.shape[1] < min_assets:
        raise PortfolioError(f"Need at least {min_assets} assets with usable return history.")
    return clean


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise correlations from cleaned simple returns."""
    clean = clean_returns(returns)
    correlation = clean.corr()
    correlation = correlation.replace([np.inf, -np.inf], np.nan)
    return correlation.reindex(index=clean.columns, columns=clean.columns)


def estimate_annualized_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualized arithmetic mean returns and covariance."""
    clean = clean_returns(returns)
    mean_returns = clean.mean() * TRADING_DAYS
    covariance = annualized_covariance(clean)
    return mean_returns, covariance


def annualized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate annualized covariance, using shrinkage when scikit-learn is available."""
    clean = clean_returns(returns)
    try:
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(clean.to_numpy(dtype=float))
        covariance = pd.DataFrame(estimator.covariance_, index=clean.columns, columns=clean.columns)
    except Exception:
        covariance = clean.cov()

    covariance = covariance.replace([np.inf, -np.inf], np.nan).fillna(0.0) * TRADING_DAYS
    cov_values = (covariance.to_numpy(dtype=float) + covariance.to_numpy(dtype=float).T) / 2
    jitter = max(float(np.trace(cov_values)) / max(len(cov_values), 1), 1e-8) * 1e-8
    cov_values = cov_values + np.eye(len(cov_values)) * jitter
    return pd.DataFrame(cov_values, index=clean.columns, columns=clean.columns)


def portfolio_stats(
    weights: pd.Series,
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    optimizer_name: str,
) -> PortfolioStats:
    """Calculate expected return, volatility, and Sharpe for a weight vector."""
    aligned_weights = weights.reindex(mean_returns.index).fillna(0.0)
    expected_return = float(aligned_weights @ mean_returns)
    volatility = float(np.sqrt(aligned_weights.T @ covariance @ aligned_weights))
    sharpe = sharpe_ratio_from_annualized(expected_return, volatility, risk_free_rate)
    return PortfolioStats(
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=float(sharpe),
        optimizer=optimizer_name,
    )


def compute_portfolio_returns(returns: pd.DataFrame, weights: pd.Series, name: str = "portfolio") -> pd.Series:
    """Apply static weights to simple asset returns."""
    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    return returns.mul(aligned_weights, axis=1).sum(axis=1).rename(name)


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Return cumulative simple returns."""
    return (1 + returns.dropna()).cumprod() - 1


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Build a capital curve from simple returns."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float, name="Equity")
    return (initial_capital * (1 + clean).cumprod()).rename("Equity")


def total_return(returns: pd.Series) -> float:
    """Calculate total compounded return."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float((1 + clean).prod() - 1)


def cagr(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Calculate compound annual growth rate from daily returns."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    ending = float((1 + clean).prod())
    if ending <= 0:
        return -1.0
    return float(ending ** (trading_days / len(clean)) - 1)


def annualized_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Calculate annualized arithmetic mean return."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean() * trading_days)


def annualized_volatility(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Calculate annualized volatility."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio."""
    ann_return = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    return sharpe_ratio_from_annualized(ann_return, ann_vol, risk_free_rate)


def sharpe_ratio_from_annualized(annual_return: float, annual_volatility: float, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from annualized return and volatility."""
    if not np.isfinite(annual_volatility) or annual_volatility <= 0:
        return np.nan
    return float((annual_return - risk_free_rate) / annual_volatility)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days: int = TRADING_DAYS) -> float:
    """Calculate annualized Sortino ratio using zero downside threshold."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    downside = clean[clean < 0]
    if len(downside) < 2:
        return np.nan
    downside_deviation = float(downside.std(ddof=1) * np.sqrt(trading_days))
    if downside_deviation <= 0:
        return np.nan
    return float((annualized_return(clean, trading_days) - risk_free_rate) / downside_deviation)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown from a simple-return series."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float, name="Drawdown")
    wealth = (1 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.rename("Drawdown")


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    drawdown = drawdown_series(returns)
    if drawdown.empty:
        return np.nan
    return float(drawdown.min())


def annual_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate calendar-year compounded returns."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float, name=getattr(returns, "name", None))
    annual = (1 + clean).groupby(clean.index.year).prod() - 1
    annual.index = annual.index.astype(str)
    annual.index.name = "Year"
    return annual


def worst_drawdown_periods(returns: pd.Series, count: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Return the worst drawdown periods for visualization."""
    drawdown = drawdown_series(returns)
    if drawdown.empty:
        return []
    underwater = drawdown < 0
    periods = []
    start = None
    previous_timestamp = drawdown.index[0]

    for timestamp, is_underwater in underwater.items():
        if is_underwater and start is None:
            start = timestamp
        elif not is_underwater and start is not None:
            end = previous_timestamp
            periods.append((start, end, float(drawdown.loc[start:end].min())))
            start = None
        previous_timestamp = timestamp

    if start is not None:
        periods.append((start, drawdown.index[-1], float(drawdown.loc[start:].min())))
    return sorted(periods, key=lambda item: item[2])[:count]


def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate historical VaR as a positive loss percentage."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cutoff = 1 - confidence_level
    return float(max(0.0, -np.quantile(clean, cutoff)))


def historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate historical CVaR as a positive loss percentage."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    cutoff_return = float(np.quantile(clean, 1 - confidence_level))
    tail = clean[clean <= cutoff_return]
    if tail.empty:
        return float(max(0.0, -cutoff_return))
    return float(max(0.0, -tail.mean()))


def compute_turnover(weights: pd.Series, previous_weights: pd.Series | None) -> float | None:
    """Calculate one-way turnover between two weight vectors."""
    if previous_weights is None or previous_weights.empty:
        return None
    aligned = pd.concat(
        [weights.rename("current"), previous_weights.rename("previous")],
        axis=1,
    ).fillna(0.0)
    return float(0.5 * (aligned["current"] - aligned["previous"]).abs().sum())


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate annualized tracking error."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate annualized information ratio."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(active) > 1 else np.nan
    excess = float(active.mean() * TRADING_DAYS)
    return excess / te if te and np.isfinite(te) and te > 0 else np.nan


def evaluate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    alpha_threshold: float = 0.0,
    information_ratio_threshold: float = 0.5,
) -> AlphaMetrics:
    """Estimate alpha/beta and information ratio using aligned daily returns."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if len(aligned) < 30:
        raise PortfolioError("Need at least 30 overlapping observations for alpha validation.")

    portfolio_values = aligned["portfolio"].to_numpy(dtype=float)
    benchmark_values = aligned["benchmark"].to_numpy(dtype=float)
    variance = np.var(benchmark_values, ddof=1)
    if np.isclose(variance, 0):
        raise PortfolioError("Benchmark returns have near-zero variance; beta cannot be estimated.")

    alpha_p_value = None
    try:
        import statsmodels.api as sm

        model_data = aligned.copy()
        x = sm.add_constant(model_data["benchmark"])
        model = sm.OLS(model_data["portfolio"], x, missing="drop").fit()
        daily_alpha = float(model.params["const"])
        beta = float(model.params["benchmark"])
        alpha_p_value = float(model.pvalues["const"])
    except Exception:
        beta = float(np.cov(portfolio_values, benchmark_values, ddof=1)[0, 1] / variance)
        daily_alpha = float(portfolio_values.mean() - beta * benchmark_values.mean())

    active_returns = aligned["portfolio"] - aligned["benchmark"]
    te = float(active_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess_return = float(active_returns.mean() * TRADING_DAYS)
    info_ratio = excess_return / te if te > 0 else np.nan
    alpha_annualized = daily_alpha * TRADING_DAYS
    passed = bool(alpha_annualized > alpha_threshold and info_ratio > information_ratio_threshold)

    return AlphaMetrics(
        alpha_annualized=float(alpha_annualized),
        beta=float(beta),
        information_ratio=float(info_ratio),
        excess_return_annualized=float(excess_return),
        tracking_error=float(te),
        observations=len(aligned),
        passed=passed,
        alpha_p_value=alpha_p_value,
    )


def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free_rate: float = 0.0) -> pd.Series:
    """Calculate rolling annualized Sharpe ratio."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float, name="Rolling Sharpe")
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = clean - daily_rf
    rolling_mean = excess.rolling(window).mean() * TRADING_DAYS
    rolling_vol = clean.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return (rolling_mean / rolling_vol).replace([np.inf, -np.inf], np.nan).rename("Rolling Sharpe")


def compute_risk_contribution(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Calculate each asset's contribution to portfolio variance."""
    aligned = weights.reindex(covariance.index).fillna(0.0).to_numpy(dtype=float)
    contribution = risk_contribution_array(aligned, covariance.to_numpy(dtype=float))
    return pd.Series(contribution, index=covariance.index, name="Risk Contribution")


def risk_contribution_array(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Calculate variance risk contribution for optimizer internals."""
    portfolio_variance = float(weights.T @ covariance @ weights)
    if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
        return np.zeros_like(weights, dtype=float)
    marginal_risk = covariance @ weights
    contribution = weights * marginal_risk / portfolio_variance
    return np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)


def performance_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Calculate the dashboard's shared performance metric set."""
    metrics = {
        "Total Return": total_return(returns),
        "CAGR": cagr(returns),
        "Annualized Return": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
        "Max Drawdown": max_drawdown(returns),
        f"VaR {int(confidence_level * 100)}%": historical_var(returns, confidence_level),
        f"CVaR {int(confidence_level * 100)}%": historical_cvar(returns, confidence_level),
    }
    if benchmark_returns is not None:
        alpha = evaluate_alpha(returns, benchmark_returns, alpha_threshold=-np.inf, information_ratio_threshold=-np.inf)
        metrics.update(
            {
                "Alpha": alpha.alpha_annualized,
                "Beta": alpha.beta,
                "Information Ratio": alpha.information_ratio,
                "Tracking Error": alpha.tracking_error,
            }
        )
    return metrics


def metrics_table(metrics_by_series: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Convert nested metric dictionaries to a display table."""
    rows = []
    for label, values in metrics_by_series.items():
        row = {"Series": label}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def gain_loss_expected_value(
    outcomes: np.ndarray | pd.Series | list[float],
    breakeven: float = 0.0,
) -> GainLossExpectedValue:
    """Calculate mean EV and gain/loss EV from simulated return or PnL outcomes."""
    try:
        values = np.asarray(outcomes, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PortfolioError("Simulation outcomes must be numeric.") from exc

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise PortfolioError("Simulation outcomes must contain at least one finite value.")
    if not np.isfinite(float(breakeven)):
        raise PortfolioError("Breakeven must be finite.")

    centered = finite_values - float(breakeven)
    breakeven_mask = np.isclose(centered, 0.0, rtol=1e-12, atol=1e-12)
    gain_mask = (centered > 0.0) & ~breakeven_mask
    loss_mask = (centered < 0.0) & ~breakeven_mask

    gains = centered[gain_mask]
    losses = centered[loss_mask]
    total = int(centered.size)
    probability_of_gain = float(gains.size / total)
    probability_of_loss = float(losses.size / total)
    probability_of_breakeven = float(breakeven_mask.sum() / total)
    average_gain = float(gains.mean()) if gains.size else 0.0
    average_loss = float(-losses.mean()) if losses.size else 0.0
    expected_value_mean = float(centered.mean())
    expected_value_gain_loss = probability_of_gain * average_gain - probability_of_loss * average_loss

    return GainLossExpectedValue(
        total_simulations=total,
        probability_of_gain=probability_of_gain,
        probability_of_loss=probability_of_loss,
        probability_of_breakeven=probability_of_breakeven,
        average_gain=average_gain,
        average_loss=average_loss,
        expected_value_mean=expected_value_mean,
        expected_value_gain_loss=float(expected_value_gain_loss),
        difference_between_methods=float(expected_value_mean - expected_value_gain_loss),
        median_outcome=float(np.median(centered)),
        best_outcome=float(np.max(centered)),
        worst_outcome=float(np.min(centered)),
    )


def calculate_risk_metrics_from_simulations(
    simulations: pd.DataFrame,
    initial_portfolio: float,
    confidence_level: float = 0.95,
) -> RiskMetrics:
    """Calculate tail-risk metrics from Monte Carlo final values."""
    if not 0 < confidence_level < 1:
        raise PortfolioError("Confidence level must be between 0 and 1.")
    if simulations.empty:
        raise PortfolioError("Simulation output cannot be empty.")

    final_values = simulations.iloc[-1]
    final_returns = final_values / initial_portfolio - 1
    final_pnl = final_values - initial_portfolio
    return_gain_loss_ev = gain_loss_expected_value(final_returns)
    pnl_gain_loss_ev = gain_loss_expected_value(final_pnl)
    mean_final_value = float(final_values.mean())
    mean_return = float(final_returns.mean())
    at_or_above_mean = (final_returns > mean_return) | np.isclose(final_returns, mean_return, rtol=1e-12, atol=1e-12)
    probability_at_or_above_mean_return = float(at_or_above_mean.mean())
    loss_returns = final_returns[final_returns < 0]
    probability_of_loss = float((final_returns < 0).mean())
    average_loss_pct = max(0.0, -float(loss_returns.mean())) if not loss_returns.empty else 0.0
    tail_cutoff = 1 - confidence_level
    percentile_return = float(np.quantile(final_returns, tail_cutoff))
    tail_returns = final_returns[final_returns <= percentile_return]
    cvar_return = float(tail_returns.mean()) if not tail_returns.empty else percentile_return
    min_return = float(final_returns.min())

    var_pct = max(0.0, -percentile_return)
    cvar_pct = max(0.0, -cvar_return)
    max_loss_pct = max(0.0, -min_return)

    return RiskMetrics(
        mean_final_value=mean_final_value,
        expected_final_value=mean_final_value,
        mean_return=mean_return,
        std_return=float(final_returns.std(ddof=1)),
        probability_at_or_above_mean_return=probability_at_or_above_mean_return,
        probability_of_loss=probability_of_loss,
        average_loss_pct=average_loss_pct,
        average_loss_value=initial_portfolio * average_loss_pct,
        var_pct=var_pct,
        cvar_pct=cvar_pct,
        var_value=initial_portfolio * var_pct,
        cvar_value=initial_portfolio * cvar_pct,
        max_loss_pct=max_loss_pct,
        max_loss_value=initial_portfolio * max_loss_pct,
        return_gain_loss_ev=return_gain_loss_ev,
        pnl_gain_loss_ev=pnl_gain_loss_ev,
    )


def risk_metrics_display_values(
    simulations: pd.DataFrame,
    initial_portfolio: float,
    risk,
) -> dict[str, float | GainLossExpectedValue]:
    """Return Monte Carlo display values, rebuilding EV fields for stale metric objects."""
    if simulations.empty:
        raise PortfolioError("Simulation output cannot be empty.")

    final_values = simulations.iloc[-1]
    final_returns = final_values / initial_portfolio - 1
    final_pnl = final_values - initial_portfolio
    return_ev = getattr(risk, "return_gain_loss_ev", gain_loss_expected_value(final_returns))
    pnl_ev = getattr(risk, "pnl_gain_loss_ev", gain_loss_expected_value(final_pnl))
    mean_return = float(getattr(risk, "mean_return", final_returns.mean()))
    at_or_above_mean = (final_returns > mean_return) | np.isclose(final_returns, mean_return, rtol=1e-12, atol=1e-12)

    return {
        "expected_final_value": float(getattr(risk, "expected_final_value", getattr(risk, "mean_final_value", final_values.mean()))),
        "mean_return": mean_return,
        "std_return": float(getattr(risk, "std_return", final_returns.std(ddof=1))),
        "probability_at_or_above_mean_return": float(
            getattr(risk, "probability_at_or_above_mean_return", at_or_above_mean.mean())
        ),
        "probability_of_loss": float(getattr(risk, "probability_of_loss", return_ev.probability_of_loss)),
        "average_loss_pct": float(getattr(risk, "average_loss_pct", return_ev.average_loss)),
        "average_loss_value": float(getattr(risk, "average_loss_value", pnl_ev.average_loss)),
        "var_value": float(getattr(risk, "var_value", np.nan)),
        "cvar_value": float(getattr(risk, "cvar_value", np.nan)),
        "max_loss_pct": float(getattr(risk, "max_loss_pct", np.nan)),
        "max_loss_value": float(getattr(risk, "max_loss_value", np.nan)),
        "return_gain_loss_ev": return_ev,
        "pnl_gain_loss_ev": pnl_ev,
    }
