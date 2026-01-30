# Sentiment-Adjusted Black–Scholes

An interpretable extension of the Black–Scholes option pricing model that incorporates **financial news sentiment** into volatility estimation using historical market data.

---

## Overview

The classical Black–Scholes model assumes that volatility is driven solely by historical price dynamics.  
In practice, **news flow and investor sentiment** play an important role in shaping market expectations and uncertainty.

This project explores a simple, stable, and explainable way to integrate **news-based sentiment** into option pricing while preserving the original Black–Scholes framework.

The result is a fully modular, end-to-end Python pipeline that compares baseline Black–Scholes prices with sentiment-adjusted prices over time.

---

## Motivation

During periods of high uncertainty (earnings, macro events, market stress), volatility is often influenced by sentiment reflected in financial news.  
Rather than replacing Black–Scholes, this project **augments volatility** using sentiment as an additional signal.

The goal is to demonstrate:
- How unstructured text data can be integrated into a classical financial model
- How sentiment can influence option prices in a controlled and interpretable way

---

## Methodology

### 1. Stock Data
- Historical NVDA stock prices (2020–2022)
- Rolling historical volatility estimation

### 2. Baseline Pricing
- European call and put pricing using the Black–Scholes formula
- Fixed risk-free rate and time-to-maturity

### 3. Sentiment Extraction
- Historical financial news from the **FNSPID** dataset (Nasdaq subset)
- Headlines filtered by ticker (`NVDA`)
- Daily sentiment scores computed using **VADER**
- Sentiment aggregated at the daily level

### 4. Sentiment-Adjusted Volatility
Volatility is adjusted using a linear sentiment scaling rule:

σ_adj = σ × (1 + α × sentiment)


Where:
- `sentiment ∈ [-1, 1]`
- `α` is a small configurable scaling factor
- A safety floor prevents negative or zero volatility

### 5. Evaluation
- Compare baseline vs sentiment-adjusted option prices
- Metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Visual comparison of option prices over time

---

## Results

- Sentiment-adjusted prices closely track baseline Black–Scholes prices under neutral conditions
- During periods of strong news sentiment, option prices diverge in a smooth and controlled manner
- The adjustment does not introduce numerical instability or unrealistic pricing behavior

A comparison plot shows baseline and sentiment-adjusted call prices over time for NVDA.

---

## Dataset

This project uses the **FNSPID: A Comprehensive Financial News Dataset in Time Series** (Nasdaq subset).

**Citation:**
Dong, Z., Fan, X., Peng, Z. (2024).
FNSPID: A Comprehensive Financial News Dataset in Time Series.
arXiv:2402.06698


**License:**  
CC BY-NC 4.0 (Non-Commercial)

> Raw dataset files are not included in this repository and are used strictly for research and educational purposes.

---

## Limitations

- Uses historical volatility rather than implied volatility
- Option prices are theoretical (not calibrated to market option chains)
- Sentiment-volatility relationship is heuristic by design

These choices are intentional to preserve interpretability and stability.

---

## Future Work

- Calibration using real option market prices
- Non-linear sentiment-volatility mappings
- Comparison with transformer-based sentiment models (e.g., FinBERT)
- Extension to multiple tickers and asset classes

---

## Tech Stack

- Python
- pandas, numpy, scipy
- nltk (VADER)
- matplotlib

---

## Author

**Vamsee Krishna Padala**