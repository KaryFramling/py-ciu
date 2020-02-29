from ciu import determine_ciu
from ciu.ciu_object import CiuObject
from ciu.ciu import _generate_samples
from random_forest import generate_model
from loan_data_generator import generate_data

data = generate_data()
train_data = data['train'][1]
test_data_encoded = data['test'][1].drop(['approved'], axis=1)

model = generate_model(train_data)

example_prediction = model.predict_proba([test_data_encoded.values[0]])
prediction_index = 0 if example_prediction[0][0] > 0.5 else 1

category_mapping = {
    'gender': ['gender_female', 'gender_male', 'gender_other'],
    'job_type': ['job_type_fixed', 'job_type_none', 'job_type_permanent']
}

feature_interactions = [{'assets_income': ['assets', 'monthly_income']}]
min_maxs = {
        'age': [20, 70, True],
        'assets': [-20000, 150000, True],
        'monthly_income': [0, 20000, True],
        'gender_female': [0, 1, True],
        'gender_male': [0, 1, True],
        'gender_other': [0, 1, True],
        'job_type_fixed': [0, 1, True],
        'job_type_none': [0, 1, True],
        'job_type_permanent': [0, 1, True]
    }

case = test_data_encoded.iloc[0, :].to_dict()
ciu = determine_ciu(
    case,
    model.predict_proba,
    min_maxs,
    1000,
    prediction_index,
    category_mapping,
    feature_interactions
)


def test_basic_ci():
    """CI should be a value between (including) 0 and 1 for each feature and
    feature combination"""
    for ci in list(ciu.ci.values()):
        assert 0 <= ci <= 1


def test_basic_cu():
    """CU should be a value between (including) 0 and 1 for each feature and
    feature combination"""
    for cu in list(ciu.cu.values()):
        assert 0 <= cu <= 1


def test_text_ciu():
    """The ``determine_ciu`` function should return a textual explanation
    for the combined CI/CU of each feature and feature combination"""
    print(ciu.interactions)
    ci_values = list(ciu.ci.values())
    cu_values = list(ciu.cu.values())
    features = list(ciu.ci.keys())
    for index, explanation in enumerate(ciu.text_explain()):
        ci = ci_values[index]
        cu = cu_values[index]
        ci_r = round(ci_values[index] * 100, 2)
        cu_r = round(cu_values[index] * 100, 2)
        importance = CiuObject._determine_importance(ci)
        typicality = CiuObject._determine_typicality(cu)
        assert explanation == f'The feature "{features[index]}", which is ' \
                               f'{importance} (CI={ci_r}%), is {typicality} ' \
                               f'for its class (CU={cu_r}%).'


def test_sample_generator_categories():
    """The sample data generator should generate data for one-hot encoded
    categories correctly: per case and per category, exactly one feature should
    be set to 1, the others should be zero."""
    samples = _generate_samples(case, min_maxs.keys(),
                                min_maxs, 1000, [0], category_mapping)

    for _, sample_case in samples.iterrows():
        for categories in category_mapping.values():
            active_categories = 0
            for category in categories:
                if sample_case[category] == 1: active_categories +=1
            assert active_categories == 1
