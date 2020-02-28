[![CircleCI](https://circleci.com/gh/TimKam/py-ciu.svg?style=svg)](https://github.com/TimKam/py-ciu)

# py-ciu

*Explainable Machine Learning through Contextual Importance and Utility*

The *py-ciu* library provides methods to generate post-hoc explanations for
machine learning-based classifiers.
It is model agnostic and answers the following questions, given a classification
decision:

* How **important** is a specific feature or feature combination for the
  classification decision? (Contextual Importance, CI)

* How **typical** is a specific feature or feature combination for the
  given class? (Contextual Utility, CU)


## Usage

Install *py-ciu*:

```
pip install py-ciu
```

Import the library:

```python
from ciu import determine_ciu
```

For the sake of the example, let us also import a data generator, create a
synthetic data set, and train a model:

```python
from sklearn.ensemble import RandomForestClassifier
from ciu_tests.loan_data_generator import generate_data

data = generate_data()
train_data = data['train'][1]
test_data = data
test_data_encoded = data['test'][1].drop(['approved'], axis=1)
random_forest = RandomForestClassifier(
    n_estimators=1000,
    random_state=42
)

labels = train_data[['approved']].values.ravel()
data = train_data.drop(['approved'], axis=1)
random_forest.fit(data, labels)
```

Then we classify the case we want to explain and determine the prediction index
for the class we are interested in:

```python
feature_names = [
    'age', 'assets', 'monthly_income', 'gender_female', 'gender_male',
    'gender_other', 'job_type_fixed', 'job_type_none', 'job_type_permanent'
]

case = test_data_encoded.values[0]
example_prediction = random_forest.predict([test_data_encoded.values[0]])
example_prediction_prob = random_forest.predict_proba([test_data_encoded.values[0]])
prediction_index = 0 if example_prediction[0] > 0.5 else 1

print(feature_names)
print(f'Case: {case}; Prediction {example_prediction}; Probability: {example_prediction_prob}')
```

Now, we can call *py-ciu*'s ``determine_ciu`` function.
The function takes the following parameters:

* ``case_data``: A dictionary that contains the data of the case.

* ``predictor``: The prediction function of the black-box model *py-ciu* should
                 call.
                 
* ``min_maxs``: A dictionary that contains the feature name (key) and a list of
                minimal, maximal value, plus a value that indicates if the value
                has to be an integer (``'feature_name': [min, max, is_int]``).

* ``samples`` (optional): The number of samples *py-ciu* will generate. Defaults
                          to ``1000``.

* ``prediction_index`` (optional): In case the model returns several
                                   predictions, it is possible to provide the
                                   index of the relevant prediction. Defaults to
                                   ``None``.
                                   
* ``category_mapping`` (optional): A mapping of one-hot encoded categorical
                                   variables to lists of categories and category
                                   names. Defaults to ``None``.
                                   
* ``feature_interactions`` (optional): A list of ``{key: list}`` tuples of
                                       features whose interactions should be
                                       evaluated. Defaults to ``[]``.


We configure the CIU parameters and call the CIU function:

```python
category_mapping = {
    'gender': ['gender_female', 'gender_male', 'gender_other'],
    'job_type': ['job_type_fixed', 'job_type_none', 'job_type_permanent']
}

feature_interactions = [{'assets_income': ['assets', 'monthly_income']}]

ciu = determine_ciu(
    test_data_encoded.iloc[0, :].to_dict(),
    random_forest.predict_proba,
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
```

The function returns a ``ciu`` object, from which we retrieve the CIU metrics:

```python
print(ciu.ci, ciu.cu)
```

We can also auto-generate CIU plots:

```python
ciu.plot_ci()
ciu.plot_cu()
```

Moreover, we can generate textual explanations based on CIU:

```python
print(ciu.text_explain())
```


Take a look at the
[examples](https://github.com/TimKam/py-ciu/tree/master/examples) directory to
learn more.

## Authors

* [Timotheus Kampik](https://github.com/TimKam/)

* [Sule Anjomshoae](https://github.com/shulemsi)

## License
The library is released under the [BSD Clause-2 License](./LICENSE).
