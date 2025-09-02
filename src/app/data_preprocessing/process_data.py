import pandas as pd
from tqdm import tqdm

from src.feature_engineering.build_labels import build_triple_barier_labels_custom  # krok 1
from src.data_preprocessing.data_preprocessor import (
    deduplicate_price_data,  # krok 2
    remove_overlapped_observations,  # krok 3
    load_data,  # krok 4
    merge_data,  # krok 5
    save_processed_data,  # krok 6
    create_abt  # krok 7
)
from src.utils.tools import (
    get_avalilable_tickers,
    filter_sp500_companies,
    calculate_financial_ratios
)


def main():
    label = 'label_1000_100_250'
    # krok 1 - buduję zmienne celu
    tickers = get_avalilable_tickers()
    for ticker in tqdm(tickers):
        build_triple_barier_labels_custom(
            ticker=ticker,
            profit_targets=[1000],  # nie stopuję algorytmu, jeśli chodzi o maksymalny zysk; będę sprzedawać akcje po ok. roku, niezależnie od zysku, czy straty
            stop_loses=[100],  # dopuszczam całkowitą stratę
            max_days=[250],  # chcę trzymać akcje ok. roku - 250 dni roboczych
            overwrite=True,
            verbose=False
        )

    # krok 2 - usuwam nadmiarowe dane dotyczące cen
    tickers = get_avalilable_tickers()
    for ticker in tqdm(tickers):
        deduplicate_price_data(ticker, verbose=False)

    # krok 3 - usuwam nachodzące na siebie labele
    tickers = get_avalilable_tickers()
    label_time = 5  # efektywnie 4 obserwacji przerwy; 1 rok (4 kwartały) + 1 kwartał marginesu
    for idx, ticker in enumerate(tqdm(tickers)):
        offset = idx % label_time
        remove_overlapped_observations(ticker, offset, label_time, verbose=False)

    tickers_bs = get_avalilable_tickers('data/raw/balance_sheets/')
    tickers_cs = get_avalilable_tickers('data/raw/company_profiles/')
    tickers_is = get_avalilable_tickers('data/raw/income_statements/')
    tickers_pr = get_avalilable_tickers('data/raw/price_history/STAGE_4/')

    for ticker in tqdm(tickers_pr):
        if (ticker in tickers_bs) & (ticker in tickers_cs) & (ticker in tickers_is):
            data = load_data(ticker)  # krok 4 - wczytanie danych
            merged_df = merge_data(data)  # krok 5 - połączenie danych z kroku 4
            save_processed_data(merged_df, ticker)  # krok 6 - zapisanie połączonych danych
    create_abt(target=label)  # krok 7 - utworzenie ABT

    # drobne korekty w ABT
    abt = pd.read_feather('data/abt/{}.feather'.format(label))
    abt_clean = filter_sp500_companies(abt)
    abt_clean = calculate_financial_ratios(abt_clean)
    abt_clean = abt_clean.sort_values(by='date', ascending=True).reset_index(drop=True)
    abt_clean.to_feather('data/abt/{}_clean.feather'.format(label))


if __name__ == "__main__":
    main()
