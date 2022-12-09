
def get_heart_disease_rf():
    """
    :return: heart disease CIU Object with a Random Forest Classifier
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from ciu.ciu_core import determine_ciu

    model = RandomForestClassifier(n_estimators=100)

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
    df.columns = ["age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                          "thalach","exang", "oldpeak","slope", "ca", "thal", "num"]

    df = df.replace({'?':np.nan}).dropna()

    df.loc[df["num"] > 0, "num"] = 1

    X = df.drop('num',axis=1)
    y = df['num']

    for i in df.columns:
        if 'object' in str(df[i].dtypes):
            df[i] = df[i].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    model.fit(X_train, y_train)

    ciu = determine_ciu(
        X_test.iloc[[23]],
        model.predict_proba,
        df.to_dict('list'),
        samples = 1000,
        prediction_index = 0
    )

    return ciu