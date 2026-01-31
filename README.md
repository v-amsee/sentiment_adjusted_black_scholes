# Sentiment-Adjusted Black–Scholes

An interpretable extension of the Black–Scholes option pricing model that incorporates **financial news sentiment** into volatility estimation and validates pricing accuracy against **real option market data**.


## Overview

The classical Black–Scholes model assumes that volatility is driven solely by historical price dynamics.  
In practice, **news flow and investor sentiment** play an important role in shaping market expectations and uncertainty.

This project explores a simple, stable, and explainable way to integrate **news-based sentiment** into option pricing while preserving the original Black–Scholes framework.

The result is a fully modular, end-to-end Python pipeline that compares baseline Black–Scholes prices with sentiment-adjusted prices and evaluates them against **real traded option contracts**.


## Motivation

During periods of high uncertainty (earnings, macro events, market stress), volatility is often influenced by sentiment reflected in financial news.  
Rather than replacing Black–Scholes, this project **augments volatility** using sentiment as an additional signal.

**Goals:**

- Integrate unstructured text data into a classical financial model  
- Demonstrate controlled and interpretable sentiment-driven volatility adjustments  
- Compare theoretical option pricing against **real market data**


## Methodology

### 1. Market Data
- Historical **NVDA stock prices (2020–2022)**
- Rolling historical volatility estimation
- **Real NVDA option chains with implied volatility**

### 2. Baseline Pricing
- European call & put pricing using the Black–Scholes formula
- Risk-free rate approximation
- **Market-aligned implied volatility**

### 3. Sentiment Extraction
- Financial headlines from the **FNSPID** dataset (Nasdaq subset)
- Headlines filtered by ticker `NVDA`
- Daily sentiment scores computed using **VADER**
- Daily aggregation of sentiment

### 4. Sentiment-Adjusted Volatility

```
σ_adj = σ × (1 + α × sentiment)
```

- `sentiment ∈ [-1, 1]`
- Small `α` ensures stability
- Volatility floor prevents negative or zero values

### 5. Evaluation
- Compare baseline vs sentiment-adjusted prices
- Validate against **real market option prices**
- Metrics:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
- Visual comparison of model vs market prices


## Results

**Evaluation on ~1,400 real NVDA option contracts (2020–2022):**

| Model                         | Call MAE | Put MAE |
|------------------------------|---------:|--------:|
| Black–Scholes (Implied Vol)  | **0.56** | **1.11** |
| Sentiment-Adjusted           | 0.60     | 1.08     |

### Key Observations

- Switching from historical to **implied volatility reduced pricing error by ~85%**
- Sentiment produced **marginal improvements for put options**
- No numerical instability or unrealistic pricing behavior observed
- Results reinforce that **implied volatility already embeds forward-looking sentiment**


## Datasets

### Financial News
**FNSPID – Financial News in Time Series (Nasdaq subset)**  
Dong, Z., Fan, X., Peng, Z. (2024). *arXiv:2402.06698*  
**License:** CC BY-NC 4.0 (Non-Commercial)

### Option Market Data
**NVDA Daily Option Chains (Q1 2020 – Q4 2022)**  
Source: Kaggle – *NVDA Daily Option Chains Dataset*  
Used for real market validation and implied volatility extraction.

> Raw dataset files are **not included** in this repository and are used strictly for research and educational purposes.


## Limitations

- Sentiment-volatility mapping is heuristic by design  
- Assumes European option framework  
- Headline-level sentiment (not full-text NLP)  
- Does not model volatility surfaces or Greeks


## Future Work

- Calibration against implied volatility surfaces  
- Non-linear sentiment-volatility mappings  
- Transformer-based sentiment models (e.g., **FinBERT**)  
- Multi-asset and cross-sector extensions


## Tech Stack

- **Python**
- pandas • numpy • scipy
- nltk (**VADER**)
- matplotlib


## Author

**Vamsee Krishna Padala**
