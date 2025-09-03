from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pyxirr import xirr


def standardize_column_names(columns: List[str]) -> List[str]:
    """Convert column names to snake_case style.

    This function lowercases all characters and replaces spaces and hyphens
    with underscores.

    Args:
        columns (List[str]): List of column names.

    Returns:
        List[str]: Standardized column names.
    """
    return [col.lower().replace(" ", "_").replace("-", "_") for col in columns]


def ensure_directory(path: Path | str) -> None:
    """Ensure that a directory exists at the given path.

    Creates the directory and any missing parent directories if they
    do not already exist.

    Args:
        path (Path | str): Path to the directory.

    Returns:
        None
   """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_available_tickers(data_dir: Path | str = "data/raw/price_history/STAGE_1") -> List[str]:
    """Return a sorted list of available tickers from feather files.

    The function scans the given directory for files with the `.feather`
    extension and extracts ticker symbols from their filenames.

    Args:
        data_dir (Path | str, optional): Directory containing feather files.
            Defaults to "data/raw/price_history/STAGE_1".

    Returns:
        List[str]: Sorted list of unique ticker symbols.
    """
    p = Path(data_dir)
    tickers = [f.stem.replace(".feather", "") if f.suffix != "" else f.stem for f in p.glob("*.feather")]
    # if filenames like "AAPL.feather" -> stem is "AAPL"
    tickers = sorted({t for t in tickers})
    return tickers


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    return numerator / denom


def calculate_portfolio_xirr(df: pd.DataFrame, X, pred, best_thresh, label, label_length=12, investment_amount=1000):
    """Simulate a trading strategy and calculate its XIRR performance.

    The function builds a simple portfolio backtest by:
    - Selecting candidate stocks whose prediction score exceeds a threshold.
    - Buying the best candidate each period (quarterly) until a max buy date.
    - Selling positions after a fixed horizon, using triple-barrier labels.
    - Tracking invested capital, cash flows, and final profit.
    - Calculating XIRR based on the resulting cash flow log.

    Args:
        df (pd.DataFrame): DataFrame with stock data and labeling columns.
        X: Features index used to align `pred` with `df`.
        pred: Predicted scores for stock selection.
        best_thresh (float): Threshold for stock selection.
        label (str): Base name of the label columns, e.g. "tb".
        label_length (int, optional): Holding horizon in months. Defaults to 12.
        investment_amount (int, optional): Amount of cash added at each buy date.
            Defaults to 1000.

    Returns:
        Dict[str, Any]: Dictionary with:
            - "log" (pd.DataFrame): Transactions log.
            - "xirr_percent" (float): Annualized return in percent.
            - "total_amount_invested" (float): Total invested capital.
            - "final_capital" (float): Portfolio final value.
            - "profit" (float): Profit = final capital - invested capital.
    """
    stocks = df.loc[X.index].copy()
    stocks = stocks.sort_values("date")
    stocks["score"] = pred
    stocks["end_date"] = stocks[f"{label}_event_date"].astype(str)
    stocks["end_adj_close"] = stocks["adj_close"] + (stocks["adj_close"] * (stocks[f"{label}_pct_change"] / 100.0))
    cols = ["date", "end_date", "ticker", f"{label}_pct_change", "adj_close", "end_adj_close"]

    candidates = stocks[stocks["score"] > best_thresh]
    if candidates.empty:
        return {"log": pd.DataFrame(), "xirr_percent": 0.0, "total_amount_invested": 0, "final_capital": 0, "profit": 0}

    candidates = candidates.loc[candidates.groupby("date")["score"].idxmax()][cols]

    # normalizacja dat do period-end (quarterly) if possible
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates["date"] = pd.PeriodIndex(candidates["date"].dt.to_period("Q")).to_timestamp(how="end").dt.date.astype(str)
    unique_dates = sorted(set(candidates["date"].tolist() + candidates["end_date"].tolist()))

    total_amount_invested = 0
    capital = 0
    wallet: Dict[str, Dict[str, Any]] = {}
    log = []

    # maksymalna granica buy date
    max_buy_date = pd.to_datetime(stocks["date"].max()) - pd.DateOffset(months=label_length)

    for todays_date in unique_dates:
        # sprzedaż
        stock_to_sell = wallet.pop(todays_date, None)
        if stock_to_sell:
            total_amount = stock_to_sell["num_of_shares"] * stock_to_sell["sell_price"]
            capital += total_amount
            log.append([todays_date, "sprzedaż", stock_to_sell["ticker"], stock_to_sell["sell_price"],
                        stock_to_sell["num_of_shares"], total_amount, capital - total_amount, capital])

        # kupno
        if todays_date in candidates["date"].values and pd.to_datetime(todays_date) <= max_buy_date:
            capital += investment_amount
            total_amount_invested += investment_amount
            stock_to_buy = candidates[candidates["date"] == todays_date].iloc[0]
            stock_ticker = stock_to_buy["ticker"]
            stock_buy_price = float(stock_to_buy["adj_close"])
            stock_sell_price = float(stock_to_buy["end_adj_close"])
            stock_sell_date = stock_to_buy["end_date"]
            num_shares = int(np.floor(capital / stock_buy_price))
            if num_shares > 0:
                total_amount = num_shares * stock_buy_price
                capital -= total_amount
                wallet[stock_sell_date] = {
                    "ticker": stock_ticker,
                    "buy_price": stock_buy_price,
                    "sell_price": stock_sell_price,
                    "num_of_shares": num_shares,
                    "buy_date": todays_date,
                }
                log.append([todays_date, "kupno", stock_ticker, stock_buy_price, num_shares, -total_amount, capital + total_amount, capital])

    log_df = pd.DataFrame(log, columns=["data", "operacja", "ticker", "cena", "liczba_sztuk", "kwota_calkowita", "stan_konta_przed", "stan_konta_po"])
    if log_df.empty:
        xirr_percent = 0.0
    else:
        log_df["data"] = pd.to_datetime(log_df["data"])
        try:
            xirr_value = xirr(log_df[["data", "kwota_calkowita"]])
            xirr_percent = round(xirr_value * 100, 2) if xirr_value is not None else 0.0
        except Exception:
            xirr_percent = 0.0

    profit = capital - total_amount_invested
    return {"log": log_df, "xirr_percent": xirr_percent, "total_amount_invested": total_amount_invested, "final_capital": capital, "profit": profit}
