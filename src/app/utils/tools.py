import os
import numpy as np
import pandas as pd
from pyxirr import xirr


def standardize_column_names(columns):
    columns = [col.lower().replace(' ', '_').replace('-', '_') for col in columns]
    return columns


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_avalilable_tickers(data_dir='data/raw/price_history/STAGE_1'):
    """
    Zczytuje listę tickerów.

    Parameters:
    - data_dir (str): Ścieżka do katalogu z plikami .feather.

    Returns:
    - list: lista tickerów.
    """
    tickers = []

    for file in os.listdir(data_dir):
        # Sprawdzenie, czy plik ma rozszerzenie .feather
        if file.endswith('.feather'):
            ticker = file[0:-8]
            tickers.append(ticker)
    tickers.sort()
    return tickers


def calculate_portfolio_xirr(df, X, pred, best_thresh, label, label_length=12, investment_amount=1000):
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


def safe_divide(numerator, denominator):
    return numerator / denominator.replace({0: np.nan})


def calculate_financial_ratios(df):
    """
    Oblicza wskaźniki analizy fundamentalnej dla danego DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame zawierający dane finansowe.

    Returns:
    pd.DataFrame: DataFrame z dodanymi kolumnami wskaźników.
    """
    # Tworzenie kopii DataFrame, aby nie modyfikować oryginału
    df = df.copy()
    df['price_to_sales'] = df['adj_close'] / (df['revenue'] / df['weighted_average_shares'])
    return df


def filter_sp500_companies(df, sp500_path='data/raw/tickers_sp500.csv'):
    """Szybsza wersja używająca set dla dużych zbiorów danych"""
    sp500 = pd.read_csv(sp500_path)
    sp500['date'] = pd.to_datetime(sp500['date'])

    # Stwórz set par (quarter, ticker)
    sp500_pairs = set()
    for _, row in sp500.iterrows():
        quarter = pd.Period(row['date'], freq='Q-DEC')
        tickers = [t.strip() for t in row['tickers'].split(',')]
        for ticker in tickers:
            sp500_pairs.add((quarter, ticker))

    # Filtruj używając vectorized operacji
    mask = df.apply(lambda row: (row['date'], row['ticker']) in sp500_pairs, axis=1)
    return df[mask]
