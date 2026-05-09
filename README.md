# TQQQ Moving Average Strategy

This repository contains the Python code for my systematic trading strategy analysis of whether a moving average trend-following strategy using QQQ signals and TQQQ execution can outperform passive buy-and-hold benchmarks on a risk-adjusted basis.

## Project summary

The strategy uses trend-following signals generated from QQQ and executes trades in TQQQ, the 3x leveraged Nasdaq-100 ETF. The framework includes exponential moving average crossovers, a long-term trend filter, a volatility regime filter, ATR-based stop-loss rules, and walk-forward out-of-sample testing.

## Main file

- `TQQQ_MA_Strategy GB 2350.py` — main Python script for data download, signal generation, backtesting, parameter sweep, and chart creation

## Strategy features

- QQQ used for signal generation
- TQQQ used for trade execution
- EMA crossover framework with a signal buffer
- 200-day trend filter and positive trend slope requirement
- Volatility-rank filter and regime-based position sizing
- ATR-based hard stop and trailing stop
- In-sample parameter selection and out-of-sample validation
- Comparison against buy-and-hold QQQ and buy-and-hold TQQQ

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## How to run

Run the script with:

```bash
python TQQQ_MA_Strategy-v3.py
```

## Expected output

The script downloads historical market data, runs the parameter sweep and backtest, prints summary performance statistics, and saves chart outputs for the analysis.

## Notes

- The script uses Yahoo Finance data through the `yfinance` package, so an internet connection is required.
- This project is for academic and research purposes only and does not constitute investment advice.
