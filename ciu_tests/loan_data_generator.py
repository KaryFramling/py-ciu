import random
import pandas as pd


def classify_case(case):
    """
    Classifies a loan approval case, using a set of rules that include explicit
    gender bias.

    :param case:
    :return: True, if loan should be approved, else False
    """

    if case['gender'] != 'male' and random.random() > 0.5:
        return False
    if case['monthly_income'] > 5000 and case['assets'] > 10000:
        return True

    return False


def generate_data(labelled=True):
    """
    Generates synthetic test/example data: loan application data
    :return: Pandas Dataframe
    """
    data = {
        'age': [],
        'gender': [],
        'assets': [],
        'job_type': [],
        'monthly_income': [],
        'approved': []
    }

    for _ in range(0,2000):
        case = {
            'age': random.randint(20, 70),
            'gender': ['female', 'male', 'other'][random.randint(0, 2)],
            'assets': random.randint(-20000, 150000),
            'job_type': ['fixed', 'permanent', 'none'][random.randint(0, 2)],
            'monthly_income': random.randint(0, 20000)
        }
        case['approved'] = classify_case(case)

        data['age'].append(case['age'])
        data['gender'].append(case['gender'])
        data['assets'].append(case['assets'])
        data['job_type'].append(case['job_type'])
        data['monthly_income'].append(case['monthly_income'])
        if labelled:
            data['approved'].append(case['approved'])

    data_df = pd.DataFrame(data=data)
    data_df_encoded = pd.get_dummies(data_df)
    return {
        'train': [
            data_df[:1000], data_df_encoded[:1000]
        ],
        'test': [
            data_df[-1000:], data_df_encoded[-1000:]
        ]
    }
