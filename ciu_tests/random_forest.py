from sklearn.ensemble import RandomForestClassifier
from ciu import determine_ciu

from loan_data_generator import generate_data


def generate_model(train_data):
    """
    :param train_data: dataframe to train on
    Generates example model to test CIU
    :return: example model
    """
    labelled_data = train_data
    random_forest = RandomForestClassifier(
        n_estimators=1000,
        random_state=42
    )

    labels = labelled_data[['approved']].values.ravel()
    data = labelled_data.drop(['approved'], axis=1)
    random_forest.fit(data, labels)
    return random_forest


data = generate_data()
train_data = data['train'][1]
test_data = data
test_data_encoded = data['test'][1].drop(['approved'], axis=1)

model = generate_model(train_data)

example_prediction = model.predict_proba([test_data_encoded.values[0]])
prediction_index = 0 if example_prediction[0][0] > 0.5 else 1

category_mapping = {
    'gender': ['gender_female', 'gender_male', 'gender_other'],
    'job_type': ['job_type_fixed', 'job_type_none', 'job_type_permanent']
}

feature_interactions = [['assets', 'monthly_income']]

ciu = determine_ciu(
    test_data_encoded.iloc[0, :].to_dict(),
    model.predict_proba,
    {
        'age': [20, 70, True],
        'assets': [-20000, 150000, True],
        'monthly_income': [0, 20000, True],
        'gender_female': [0, 1, True],
        'gender_male': [0, 1, True],
        'gender_other': [0, 1, True],
        'job_type_fixed': [0, 1, True],
        'job_type_none': [0, 1, True],
        'job_type_permanent': [0, 1, True]
    },
    1000,
    prediction_index,
    category_mapping,
    feature_interactions
)

print(data['test'][0].values[0])
print(test_data_encoded.values[0])
print(model.predict_proba([test_data_encoded.values[0]]))

print(ciu.ci, ciu.cu)

ciu.plot_ci()
ciu.plot_cu()

print(ciu.text_explain())

"""
TODO:
* normalize
* examples
* feature interaction
* test
* brush up
* Contrastive:
    * Take a case
    * Compute CIU
    * Always for one feature or for one set of features
    * Simulate an example with a different prediction category (rejection)
    * Compare cases: we highlight the features that are
    * Highlight required feature difference
    * Use sum of CI_case_1, CI_case_2 to highlight importance
    * Use sum of CU_case_1, CU_case_2 to highlight utility
"""

