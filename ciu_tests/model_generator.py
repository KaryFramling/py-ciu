from sklearn.ensemble import RandomForestClassifier
from ciu import determine_ciu

from data_generator import generate_data


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
    [
        [20, 70, True], [-20000, 150000, True], [0, 20000, True],
        [0, 1, True], [0, 1, True], [0, 1, True],
        [0, 1, True], [0, 1, True], [0, 1, True]
    ],
    ['age', 'assets', 'monthly_income', 'gender_female', 'gender_male',
        'gender_other', 'job_type_fixed', 'job_type_none', 'job_type_permanent'],
    1000,
    prediction_index,
    category_mapping
)
print(ciu.ci, ciu.cu)

ciu.plot_ci()

print(ciu.text_explain())

"""
TODOs:
* Improve performance
* CIU for feature interactions
* Clustering example
* Tests
* Documentation
* Brush up examples
* Publish
--------
* Third example
* Contrastive explanations
* Text explanation support
"""

