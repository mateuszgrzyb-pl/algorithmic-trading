from app.utils.tools import standardize_column_names


def test_standardize_column_names_basic():
    cols = ["First Name", "Last-Name", "Age"]
    expected = ["first_name", "last_name", "age"]
    assert standardize_column_names(cols) == expected


def test_standardize_column_names_empty_list():
    assert standardize_column_names([]) == []


def test_standardize_column_names_already_snake_case():
    cols = ["first_name", "last_name"]
    expected = ["first_name", "last_name"]
    assert standardize_column_names(cols) == expected


def test_standardize_column_names_mixed_case_and_symbols():
    cols = ["Some Column", "Another-Column", "X Y-Z"]
    expected = ["some_column", "another_column", "x_y_z"]
    assert standardize_column_names(cols) == expected
