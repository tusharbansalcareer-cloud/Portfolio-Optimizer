from __future__ import annotations

TRADING_DAYS = 252

DEFAULT_BENCHMARK = "^NSEI"
DEFAULT_RISK_FREE_RATE = 0.06
DEFAULT_TRANSACTION_COST = 0.002
DEFAULT_SLIPPAGE = 0.001
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_MAX_WEIGHT = 0.4
DEFAULT_LOOKBACK_DAYS = 504

STRATEGY_MEAN_VARIANCE = "Mean-Variance / Max Sharpe"
STRATEGY_MEAN_VARIANCE_LEGACY = "Mean-Variance Optimization"
STRATEGY_MINIMUM_VARIANCE = "Minimum Variance"
STRATEGY_MINIMUM_VARIANCE_LEGACY = "Minimum Variance Portfolio"
STRATEGY_RISK_PARITY = "Risk Parity"
STRATEGY_EQUAL_WEIGHT = "Equal Weight"
STRATEGY_CUSTOM_MANUAL = "Custom Manual Allocation"

OPTIMIZATION_STRATEGIES = (
    STRATEGY_MEAN_VARIANCE,
    STRATEGY_MINIMUM_VARIANCE,
    STRATEGY_RISK_PARITY,
    STRATEGY_EQUAL_WEIGHT,
)

STRATEGY_ALIASES = {
    STRATEGY_MEAN_VARIANCE: STRATEGY_MEAN_VARIANCE,
    STRATEGY_MEAN_VARIANCE_LEGACY: STRATEGY_MEAN_VARIANCE,
    "Max Sharpe": STRATEGY_MEAN_VARIANCE,
    "Mean-Variance": STRATEGY_MEAN_VARIANCE,
    STRATEGY_MINIMUM_VARIANCE: STRATEGY_MINIMUM_VARIANCE,
    STRATEGY_MINIMUM_VARIANCE_LEGACY: STRATEGY_MINIMUM_VARIANCE,
    STRATEGY_RISK_PARITY: STRATEGY_RISK_PARITY,
    STRATEGY_EQUAL_WEIGHT: STRATEGY_EQUAL_WEIGHT,
    STRATEGY_CUSTOM_MANUAL: STRATEGY_CUSTOM_MANUAL,
}

REBALANCE_WEEKLY = "Weekly"
REBALANCE_MONTHLY = "Monthly"
REBALANCE_QUARTERLY = "Quarterly"
SUPPORTED_REBALANCE_FREQUENCIES = (
    REBALANCE_WEEKLY,
    REBALANCE_MONTHLY,
    REBALANCE_QUARTERLY,
)

REBALANCE_PERIOD_ALIASES = {
    REBALANCE_WEEKLY: "W",
    REBALANCE_MONTHLY: "M",
    REBALANCE_QUARTERLY: "Q",
}

NAME_TO_YAHOO_TICKER = {
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "apollo hospitals": "APOLLOHOSP.NS",
    "asian paints": "ASIANPAINT.NS",
    "axis bank": "AXISBANK.NS",
    "bajaj auto": "BAJAJ-AUTO.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "bank nifty": "^NSEBANK",
    "banknifty": "^NSEBANK",
    "bharat electronics": "BEL.NS",
    "bharat petroleum": "BPCL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "cipla": "CIPLA.NS",
    "coal india": "COALINDIA.NS",
    "dr reddy": "DRREDDY.NS",
    "eicher motors": "EICHERMOT.NS",
    "eternal": "ETERNAL.NS",
    "grasim": "GRASIM.NS",
    "hcl tech": "HCLTECH.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc life": "HDFCLIFE.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "hindalco": "HINDALCO.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "icici bank": "ICICIBANK.NS",
    "indusind bank": "INDUSINDBK.NS",
    "infosys": "INFY.NS",
    "itc": "ITC.NS",
    "jio financial": "JIOFIN.NS",
    "jsw steel": "JSWSTEEL.NS",
    "kotak bank": "KOTAKBANK.NS",
    "larsen toubro": "LT.NS",
    "larsen and toubro": "LT.NS",
    "lt": "LT.NS",
    "mahindra": "M&M.NS",
    "mahindra and mahindra": "M&M.NS",
    "m and m": "M&M.NS",
    "maruti": "MARUTI.NS",
    "nifty": "^NSEI",
    "nifty 50": "^NSEI",
    "nifty50": "^NSEI",
    "ntpc": "NTPC.NS",
    "ongc": "ONGC.NS",
    "power grid": "POWERGRID.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "relience": "RELIANCE.NS",
    "sbi": "SBIN.NS",
    "sbi life": "SBILIFE.NS",
    "shriram finance": "SHRIRAMFIN.NS",
    "state bank of india": "SBIN.NS",
    "sun pharma": "SUNPHARMA.NS",
    "tata consumer": "TATACONSUM.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "tcs": "TCS.NS",
    "tech mahindra": "TECHM.NS",
    "titan": "TITAN.NS",
    "trent": "TRENT.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "wipro": "WIPRO.NS",
}

for yahoo_ticker in tuple(NAME_TO_YAHOO_TICKER.values()):
    if yahoo_ticker.endswith(".NS"):
        NAME_TO_YAHOO_TICKER.setdefault(yahoo_ticker.removesuffix(".NS").lower(), yahoo_ticker)

