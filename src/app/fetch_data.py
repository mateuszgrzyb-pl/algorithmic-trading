import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.tools import get_avalilable_tickers
from data_preprocessing.data_loader import (
    download_price_history,
    download_balance_sheets,
    download_income_statements,
    download_company_profiles
)


def main():
    tickers_df = pd.read_csv('data/raw/tickers_sp500.csv')
    tickers_df = tickers_df['tickers'].drop_duplicates()
    tickers = []
    for row in tickers_df:
        for ticker in row.split(','):
            ticker = ticker.replace(' ', '')
            tickers.append(ticker)
    tickers = np.unique(tickers)
    available_tickers = get_avalilable_tickers()
    tickers_to_download = [ticker for ticker in tickers if ticker not in available_tickers]
    for ticker in tqdm(tickers_to_download, desc='Trwa pobieranie danych'):
        start_date = '1985-01-01'
        end_date = '2024-09-30'
        api_key = os.getenv('finance_toolkit_key')

        download_price_history([ticker], start_date, end_date, api_key)
        download_balance_sheets([ticker], start_date, end_date, api_key)
        download_income_statements([ticker], start_date, end_date, api_key)
        download_company_profiles([ticker], api_key)
