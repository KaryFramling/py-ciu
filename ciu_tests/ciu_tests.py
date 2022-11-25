import pandas as pd
import numpy as np

from ciu.ciu_core import determine_ciu

def get_iris_test():
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn import datasets

    iris=datasets.load_iris()

    df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                  columns = iris['feature_names'] + ['target'])
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

    X = df[['s_length', 's_width', 'p_length', 'p_width']]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    iris_df = df.apply(pd.to_numeric, errors='ignore')

    #Can also be written manually:
    #   test_iris = pd.DataFrame.from_dict({'s_length' : [2], 's_width' : [3.2], 'p_length': [1.8], 'p_width' : [2.4]})

    ciu = determine_ciu(
        X_test.iloc[[42]],
        model.predict_proba,
        iris_df.to_dict('list'),
        samples = 1000,
        prediction_index = 2
    )

    return ciu

def get_boston_gbm_test():

    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    data = pd.DataFrame(data)
    data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    xgb.DMatrix(data=data,label=target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=123)
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg.fit(X_train,y_train)

    ciu = determine_ciu(
        X_test.iloc[[124]],
        xg_reg.predict,
        data.to_dict('list'),
        samples = 1000,
        prediction_index = None
    )

    return ciu

def get_heart_disease_rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

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
