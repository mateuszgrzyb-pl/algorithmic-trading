import pandas as pd
from app.feature_engineering.labeling import triple_barrier_labeling_custom


def build_triple_barier_labels_custom(ticker, profit_targets, stop_loses, max_days, overwrite=False, verbose=False):
    """
    Funkcja hurtowo dodaje labele do danych cenowych i zapisuje nowe pliki z etykietami.

    Parameters:
    - ticker (str): Symbol ticker’a.
    - profit_targets (list of float): Lista procentowych celów zysku.
    - stop_loses (list of float): Lista procentowych limitów straty.
    - max_days (list of int): Lista maksymalnych liczby dni do etykietowania.
    - overwrite (bool): Czy nadpisać istniejące pliki z etykietami. Domyślnie False.
    - verbose (bool): Czy wyświetlać komunikaty informacyjne. Domyślnie False.

    Returns:
    - None
    """
    # Ścieżki do plików
    input_path = f'data/raw/price_history/STAGE_1/{ticker}.feather'
    output_path = f'data/raw/price_history/STAGE_2/{ticker}.feather'

    # Wczytanie danych
    data = pd.read_feather(input_path)

    # Sprawdzenie, czy należy przetworzyć ticker
    if overwrite:
        for n in range(len(profit_targets)):
            profit_target = profit_targets[n]
            stop_loss = stop_loses[n]
            max_day = max_days[n]

            label_prefix = f'label_{profit_target}_{stop_loss}_{max_day}'

            # Dodanie etykiet za pomocą zaktualizowanej funkcji
            data = triple_barrier_labeling_custom(
                df=data,
                price_col='adj_close',
                label_name=label_prefix,
                date_col='date',
                profit_target=profit_target,
                stop_loss=stop_loss,
                max_days=max_day
            )
        # Zapisanie danych z etykietami
        data.to_feather(output_path)

        if verbose:
            print(f'Zapisano labele dla {ticker} do {output_path}')
    else:
        if verbose:
            print(f'Etykiety dla {ticker} już istnieją. Użyj overwrite=True, aby nadpisać.')
