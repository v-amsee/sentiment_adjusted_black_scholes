import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate European call option price using Black-Scholes formula.
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate European put option price using Black-Scholes formula.
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
