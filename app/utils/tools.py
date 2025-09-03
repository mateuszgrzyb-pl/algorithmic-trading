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
    max_buy_date = (stocks['date'].max().to_timestamp(how='end') - pd.DateOffset(months=label_length))
    stocks['score'] = pred
    stocks['end_date'] = stocks[f'{label}_event_date'].astype(str)
    stocks['end_adj_close'] = stocks.adj_close + (stocks.adj_close * (stocks[f'{label}_pct_change']/100))
    cols = ['date', 'end_date', 'ticker', f'{label}_pct_change', 'adj_close', 'end_adj_close']
    stocks = stocks.loc[stocks['score'] > best_thresh]
    stocks = stocks.loc[stocks.groupby('date')['score'].idxmax()][cols]
    stocks['date'] = stocks['date'].astype(str)
    stocks['date'] = pd.PeriodIndex(stocks['date'], freq='Q').to_timestamp(how='end').date.astype(str)
    unique_dates = np.unique(stocks['date'].tolist() + stocks['end_date'].tolist())
    total_amount_invested = 0
    capital = 0
    wallet = {}
    log = []

    for i, todays_date in enumerate(unique_dates):
        # 1. Sprzedaż akcji (jeśli możliwa).
        stock_to_sell = wallet.get(todays_date)  # pobieram akcje do sprzedania
        if stock_to_sell is not None:  # gdy są jakieś akcje do sprzedania
            # 1.1. Pobieram akcje.
            stock_ticker = stock_to_sell['ticker']
            stock_buy_price = stock_to_sell['buy_price']
            stock_sell_price = stock_to_sell['sell_price']
            stock_num_of_shares = stock_to_sell['num_of_shares']

            # 1.2. Aktualizuję stan konta.
            total_amount = stock_num_of_shares * stock_sell_price
            capital = capital + total_amount

            # 1.3. Usuwam akcje z portfela.
            del wallet[todays_date]

            # dodanie loga
            log.append([todays_date, 'sprzedaż', stock_ticker, stock_sell_price, stock_num_of_shares, total_amount, capital-total_amount, capital])

        # 2. Kupno akcji.
        if (todays_date in stocks['date'].tolist()) & (pd.to_datetime(todays_date) <= max_buy_date):
            capital += investment_amount  # dodaję kapitał
            total_amount_invested += investment_amount

            # 2.1. Pobieram akcje.
            stock_to_buy = stocks[stocks['date'] == todays_date]
            stock_ticker = stock_to_buy['ticker'].values[0]
            stock_buy_price = stock_to_buy['adj_close'].values[0]
            stock_sell_price = stock_to_buy['end_adj_close'].values[0]
            stock_sell_date = stock_to_buy['end_date'].values[0]
            stock_num_of_shares = int(np.floor(capital/stock_buy_price))

            if stock_num_of_shares > 0:
                # 2.2. Aktualizuję stan konta.
                total_amount = stock_num_of_shares * stock_buy_price
                capital = capital - total_amount

                # 2.3. Dodaje akcje do portfela.
                wallet[stock_sell_date] = {
                    'ticker': stock_ticker,
                    'buy_price': stock_buy_price,
                    'sell_price': stock_sell_price,
                    'num_of_shares': stock_num_of_shares,
                    'buy_date': todays_date}

                # dodanie loga
                log.append([todays_date, 'kupno', stock_ticker, stock_buy_price, stock_num_of_shares, -total_amount, capital+total_amount, capital])

    log = pd.DataFrame(log, columns=['data', 'operacja', 'ticker', 'cena', 'liczba_sztuk', 'kwota_calkowita', 'stan_konta_przed', 'stan_konta_po'])
    if log.empty:
        srednioroczny_zwrot = 0
    else:
        log['data'] = pd.to_datetime(log['data'])
        xirr_value = xirr(log[['data', 'kwota_calkowita']])
        if xirr_value is not None:
            srednioroczny_zwrot = np.round(xirr_value * 100, 2)
        else:
            srednioroczny_zwrot = 0.0
    profit = capital - total_amount_invested
    return {
        'log': log,
        'xirr_percent': srednioroczny_zwrot,
        'total_amount_invested': total_amount_invested,
        'final_capital': capital,
        'profit': profit
    }
