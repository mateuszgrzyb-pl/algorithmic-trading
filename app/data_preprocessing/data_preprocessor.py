import os
import pandas as pd

from src.utils.tools import get_avalilable_tickers


def load_data(ticker):
    """
    Wczytuje wszystkie dane dla danego tickera i zwraca je jako słownik DataFrame'ów.
    """
    data = {}
    base_path = 'data/raw/'

    paths = {
        'price_history': f'{base_path}price_history/STAGE_4/{ticker}.feather',
        'balance_sheets': f'{base_path}balance_sheets/{ticker}.feather',
        'income_statements': f'{base_path}income_statements/{ticker}.feather',
        'company_profiles': f'{base_path}company_profiles/{ticker}.feather',
    }

    for key, path in paths.items():
        if os.path.exists(path):
            df = pd.read_feather(path)
            if 'date' in df.columns:
                df['date'] = pd.PeriodIndex(df['date'], freq='Q-DEC')
            data[key] = df
        else:
            print(f'Plik {path} nie istnieje.')
    return data


def merge_data(data):
    """
    Łączy dane z różnych źródeł w jeden DataFrame.
    """
    merged_df = data['price_history']

    if 'balance_sheets' in data:
        data['balance_sheets']['date'] = data['balance_sheets']['date'] + 1
        merged_df = merged_df.merge(data['balance_sheets'], on='date', how='inner')
    if 'income_statements' in data:
        data['income_statements']['date'] = data['income_statements']['date'] + 1
        merged_df = merged_df.merge(data['income_statements'], on='date', how='inner')
    if 'company_profiles' in data:
        profile = data['company_profiles']
        for col in profile.columns:
            merged_df[col] = profile[col].iloc[0]
    return merged_df


def save_processed_data(df, ticker, verbose=False):
    """
    Zapisuje przetworzone dane do pliku Feather.
    """
    output_path = f'data/processed/{ticker}.feather'
    df.to_feather(output_path)
    if verbose:
        print(f'Przetworzone dane dla {ticker} zapisane w {output_path}')


def create_abt(target):
    """
    Przygotowanie zbioru ABT dla wybranego labela.
    """
    datas = []
    tickers = get_avalilable_tickers('data/processed/')

    # wczytanie dostępnych danych
    for ticker in tickers:
        data = pd.read_feather(f'data/processed/{ticker}.feather')
        data['ticker'] = ticker
        datas.append(data.copy(deep=True))
    data = pd.concat(datas)

    # usunięcie zbędnych kolumn
    labels = [col for col in data.columns if 'label' in col]
    labels_to_drop = [col for col in labels if target not in col]
    labels_to_drop.append('company_name')
    data = data.drop(columns=labels_to_drop)
    data.to_feather(f'data/abt/{target}.feather')


def deduplicate_price_data(ticker, verbose=False):
    """
    Funkcja czyści dane związane z cenami z nadmiarowych obserwacji, które były niezbędnę do budowy zmiennych celu i zmiennych momentum.
    """
    data = pd.read_feather(f'data/raw/price_history/STAGE_2/{ticker}.feather')
    data['date_ts'] = data['date'].dt.to_timestamp()
    data = data.sort_values('date_ts').reset_index(drop=True)
    data['quarter'] = data['date_ts'].dt.to_period('Q')
    data = data.drop_duplicates(subset='quarter', keep='last').drop(['quarter'], axis=1).reset_index(drop=True)
    data['date'] = data['date_ts'].dt.to_period(freq='Q')
    data.drop(columns=['date_ts'], inplace=True)
    data.to_feather(f'data/raw/price_history/STAGE_3/{ticker}.feather')
    if verbose:
        print(f'Wyczyszczono dane dla {ticker}')


def remove_overlapped_observations(ticker, offset, label_time, verbose=False):
    """
    Usunięcie obserwacji w których zmienne celu na siebie nachodzą.
    """
    data = pd.read_feather(f'data/raw/price_history/STAGE_3/{ticker}.feather')
    data = data.iloc[offset::label_time]
    data = data.reset_index(drop=True)
    data.to_feather(f'data/raw/price_history/STAGE_4/{ticker}.feather')
    if verbose:
        print(f'Wyczyściłem dane dla {ticker}')
