import random
import pandas as pd


def generate_data():
    """
    Generates synthetic test/example data: songs in a music library
    :return: Pandas Dataframe
    """
    data = {
        'genre': [],
        'length': [],
        'volume': [],
        'danceability': [],
        'bpm': [],
        'year': []
    }

    cases = []
    for _ in range(0,200):
        case = {
            'genre': ['rock', 'pop', 'hiphop', 'soul', 'jazz', 'classic'][random.randint(0, 5)],
            'length': random.randint(90, 300),
            'volume': random.randint(0, 10),
            'danceability': random.randint(0, 10),
            'bpm': random.randint(60, 220),
            'year': random.randint(1950, 2020)
        }
        cases.append(case)

    data_df = pd.DataFrame(cases)
    data_df_encoded = pd.get_dummies(data_df)
    return {
        'train': [
            data_df[:100], data_df_encoded[:100]
        ],
        'test': [
            data_df[-100:], data_df_encoded[-100:]
        ]
    }

