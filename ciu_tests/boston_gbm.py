
def get_boston_gbm_test():
    """
    :return: boston gbm CIU Object
    """

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from ciu.ciu_core import determine_ciu
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
