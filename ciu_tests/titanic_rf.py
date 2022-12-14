def get_titanic_rf():
    """
    Random forest model using the intermediate concepts on the Titanic dataset
    :return: Titanic CIU object with intermediate concepts
    """

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from ciu.ciu_core import determine_ciu

    data = pd.read_csv("https://raw.githubusercontent.com/KaryFramling/py-ciu/master/ciu_tests/data/titanic.csv")
    data = data.drop(data.columns[0], axis=1)
    unused = ['PassengerId','Cabin','Name','Ticket']

    for col in unused:
        data = data.drop(col, axis=1)

    from sklearn.preprocessing import LabelEncoder
    data = data.dropna().apply(LabelEncoder().fit_transform)
    train = data.drop('Survived', axis=1)

    # Create test instance (8-year old boy)
    new_passenger = pd.DataFrame.from_dict({"Pclass" : [1], "Sex": [1], "Age": [8], "SibSp": [0], "Parch": [0], "Fare": [72], "Embarked": [2]})

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train, data.Survived)

    intermediate_tit = [
            {"Wealth":['Pclass', 'Fare']},
            {"Family":['SibSp', 'Parch']},
            {"Gender":['Sex']},
            {"Age_years":['Age']},
            {"Embarked_Place":['Embarked']}
        ]

    ciu_titanic = determine_ciu(
        new_passenger,
        model.predict_proba,
        train.to_dict('list'),
        samples = 1000,
        prediction_index = 1,
        intermediate_concepts=intermediate_tit
    )

    return ciu_titanic