
def get_heart_disease_rf(inst_ind=0):
    """
    :return: heart disease CIU Object with a Random Forest Classifier
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    import ciu as ciu
    from ciu.CIU import CIU

    model = RandomForestClassifier(n_estimators=100)

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
    df.columns = ["age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                          "thalach","exang", "oldpeak","slope", "ca", "thal", "num"]

    df = df.replace({'?':np.nan}).dropna()

    df.loc[df["num"] > 0, "num"] = 1

    # Shortcut here: everything to float
    for i in df.columns:
        if 'object' in str(df[i].dtypes):
            df[i] = df[i].astype(float)

    X = df.drop('num',axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    model.fit(X_train, y_train)

    CIU = CIU(model.predict_proba, ['No', 'Yes'], data=X_train)

    instance = X_test.iloc[[inst_ind]]

    return CIU, model, instance