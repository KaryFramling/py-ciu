from sklearn.ensemble import RandomForestClassifier
from ciu import determine_ciu

from data_generator import generate_data


def generate_model():
    """
    Generates example model to test CIU
    :return: example model
    """
    labelled_data = generate_data()['train'][1]
    random_forest = RandomForestClassifier(
        n_estimators=1000,
        random_state=42
    )

    labels = labelled_data[['approved']].values.ravel()
    data = labelled_data.drop(['approved'], axis=1)
    random_forest.fit(data, labels)
    return random_forest


test_data = generate_data()['test'][0].drop(['approved'], axis=1)
test_data_encoded = generate_data()['test'][1].drop(['approved'], axis=1)

model = generate_model()

print(model.predict_proba([test_data_encoded.values[0]]))

example_prediction = model.predict_proba([test_data_encoded.values[0]])
prediction_index = 0 if example_prediction[0][0] > 0.5 else 1

category_mapping = {
    'gender': ['gender_female', 'gender_male', 'gender_other'],
    'job_type': ['job_type_fixed', 'job_type_none', 'job_type_permanent']
}

ciu = determine_ciu(
    test_data_encoded.values[0],
    model,
    test_data_encoded[:1000],
    ['age', 'assets', 'monthly_income', 'gender_female', 'gender_male',
        'gender_other', 'job_type_fixed', 'job_type_none', 'job_type_permanent'],
    prediction_index,
    category_mapping
)
print(ciu)

"""
TODOs:
* Category mapping support
* Visualization support
* Text explanation support
* Clustering example
* Tests
* Documentation
* Brush up examples
* Publish
"""

