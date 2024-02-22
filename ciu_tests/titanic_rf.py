def get_titanic_rf():
    """
    Random forest model using the intermediate concepts on the Titanic dataset
    :return: Titanic CIU object with intermediate concepts
    """

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import ciu as ciu
    from ciu.CIU import CIU

    data = pd.read_csv("https://raw.githubusercontent.com/KaryFramling/py-ciu/master/ciu_tests/data/titanic.csv")
    data = data.drop(data.columns[0], axis=1)
    unused = ['PassengerId','Cabin','Name','Ticket']

    for col in unused:
        data = data.drop(col, axis=1)

    from sklearn.preprocessing import LabelEncoder
    data = data.dropna().apply(LabelEncoder().fit_transform)
    train = data.drop('Survived', axis=1)

    # Create test instance (8-year old boy)
    new_passenger = pd.DataFrame.from_dict({"Pclass" : [1], "Sex": [1], "Age": [8.0], "SibSp": [0], "Parch": [0], "Fare": [72.0], "Embarked": [1]})

    model = RandomForestClassifier(n_estimators=100)
    model.fit(train, data.Survived)

    category_mapping = {
        'Sex': ['female','male'],
        'Pclass': list(range(max(data.Pclass))),
        'SibSp': list(range(max(data.SibSp))),
        'Parch': list(range(max(data.Parch))),
        'Embarked': ["Belfast","Cherbourg","Queenstown","Southampton"]
    }

    titanic_voc = {
        "Wealth":['Pclass', 'Fare'],
        "Family":['SibSp', 'Parch'],
        "Gender":['Sex'],
        "Age":['Age'],
        "Embarked":['Embarked']
    }

    CIU_titanic = CIU(model.predict_proba, ['No', 'Yes'], data=train, category_mapping=category_mapping, vocabulary=titanic_voc)

    return CIU_titanic, model, new_passenger