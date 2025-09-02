import pandas as pd


def triple_barrier_labeling_custom(df, price_col, label_name, date_col=None, profit_target=40, stop_loss=20, max_days=250):
    """
    Implementacja Triple Barrier Method dostosowana do specyficznych założeń, z dodatkowymi kolumnami:
    - Data osiągnięcia zdarzenia (stop loss, profit target lub data po max_days).
    - Procentowa zmiana w cenie w momencie osiągnięcia zdarzenia.

    Parameters:
    - df (pd.DataFrame): DataFrame z kolumną cenową oraz opcjonalnie kolumną daty.
    - price_col (str): Nazwa kolumny z ceną.
    - label_name (str): Nazwa / prefiks nowej zmiennej.
    - date_col (str, optional): Nazwa kolumny z datą. Jeśli None, zakłada się, że DataFrame ma indeks datowy.
    - profit_target (float): Procentowy cel zysku (domyślnie 40 dla 40%).
    - stop_loss (float): Procentowy limit straty (domyślnie 20 dla 20%).
    - max_days (int): Maksymalna liczba dni handlowych do etykietowania (domyślnie 250).

    Returns:
    - pd.DataFrame: DataFrame z dodanymi kolumnami:
        - '{label_name}_target': Zmienna celu z wartościami {1, -1, 0}.
        - '{label_name}_final_price': Cena końcowa, która spowodowała przypisanie etykiety.
        - '{label_name}_days_to_event': Liczba dni do zdarzenia (250 dni dla zysku lub brak zdarzenia, lub liczba dni do osiągnięcia stop loss).
        - '{label_name}_event_date': Data osiągnięcia zdarzenia (stop loss, profit target lub data po max_days).
        - '{label_name}_pct_change': Procentowa zmiana w momencie osiągnięcia zdarzenia.
    """
    profit_target /= 100
    stop_loss /= 100
    n = len(df)

    # Inicjalizacja list dla nowych kolumn
    labels = [0] * n
    final_prices = df[price_col].values.copy()
    days_to_events = [max_days] * n
    event_dates = [pd.NaT] * n
    pct_changes = [0.0] * n

    prices = df[price_col].values
    dates = df[date_col].dt.to_timestamp().values

    for i in range(n):
        current_price = prices[i]
        stop_triggered = False

        # Sprawdzenie warunku stop loss w trakcie max_days
        for j in range(1, min(max_days + 1, n - i)):
            future_price_j = prices[i + j]
            change_j = (future_price_j - current_price) / current_price
            if change_j <= -stop_loss:
                labels[i] = -1
                final_prices[i] = future_price_j
                days_to_events[i] = j
                event_dates[i] = dates[i + j]
                pct_changes[i] = change_j * 100
                stop_triggered = True
                break

        # Sprawdzenie wzrostu po max_days, jeśli stop loss nie wystąpił
        if not stop_triggered and i + max_days < n:
            future_price = prices[i + max_days]
            change = (future_price - current_price) / current_price
            labels[i] = 1 if change >= profit_target else 0
            final_prices[i] = future_price
            days_to_events[i] = max_days
            event_dates[i] = dates[i + max_days]
            pct_changes[i] = change * 100

        # Obsługa przypadku, gdy nie ma wystarczającej liczby dni
        elif not stop_triggered:
            labels[i] = 0  # Użyj 0 zamiast None
            final_prices[i] = prices[-1]
            days_to_events[i] = n - i - 1
            event_dates[i] = dates[-1]
            pct_changes[i] = ((prices[-1] - current_price) / current_price) * 100

    # Dodanie nowych kolumn do DataFrame
    df[f'{label_name}_target'] = labels
    df[f'{label_name}_final_price'] = final_prices
    df[f'{label_name}_days_to_event'] = days_to_events
    df[f'{label_name}_event_date'] = event_dates
    df[f'{label_name}_pct_change'] = pct_changes
    return df
