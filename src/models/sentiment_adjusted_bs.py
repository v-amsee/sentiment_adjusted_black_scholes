def adjust_volatility(
    base_volatility: float,
    sentiment: float,
    alpha: float
) -> float:
    """
    Adjust volatility based on sentiment signal.
    """
    adjusted_vol = base_volatility * (1 + alpha * sentiment)

    # Safety floor to avoid negative or zero volatility
    return max(adjusted_vol, 1e-4)
