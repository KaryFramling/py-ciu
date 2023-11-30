
def get_boston_gbm_test(inst_ind=1):
    """
    :return: CIU, XGB and instance
    """

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    import ciu as ciu
    from ciu.CIU import CIU

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    data = pd.DataFrame(data)
    data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    #xgb.DMatrix(data=data,label=target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=123)
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg.fit(X_train,y_train)

    out_minmaxs = pd.DataFrame({'mins': [min(y_train)], 'maxs': max(y_train)})
    out_minmaxs.index = ['Price']
    CIU = CIU(xg_reg.predict, ['Price'], data=X_train, out_minmaxs=out_minmaxs)

    inst_ind = inst_ind
    instance = X_test.iloc[[inst_ind]]

    return CIU, xg_reg, instance
