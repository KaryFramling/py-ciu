def get_ames_gbm_test():
    """
    :return: A CIU object and the list of intermediate concepts used in the example.
    """
    from ciu.ciu_core import determine_ciu
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('https://raw.githubusercontent.com/KaryFramling/py-ciu/master/ciu_tests/data/AmesHousing.csv')

    #Checking for missing data
    missing_data_count = df.isnull().sum()
    missing_data_percent = df.isnull().sum() / len(df) * 100

    missing_data = pd.DataFrame({
        'Count': missing_data_count,
        'Percent': missing_data_percent
    })

    missing_data = missing_data[missing_data.Count > 0]
    missing_data.sort_values(by='Count', ascending=False, inplace=True)

    #This one has spaces for some reason
    df.columns = df.columns.str.replace(' ', '')


    #Taking care of missing values
    from sklearn.impute import SimpleImputer
    # Group 1:
    group_1 = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
        'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    df[group_1] = df[group_1].fillna("None")

    # Group 2:
    group_2 = [
        'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]

    df[group_2] = df[group_2].fillna(0)

    # Group 3:
    group_3a = [
        'Functional', 'MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st',
        'Exterior2nd', 'SaleType', 'Utilities'
    ]

    imputer = SimpleImputer(strategy='most_frequent')
    df[group_3a] = pd.DataFrame(imputer.fit_transform(df[group_3a]), index=df.index)

    df.LotFrontage = df.LotFrontage.fillna(df.LotFrontage.mean())
    df.GarageYrBlt = df.GarageYrBlt.fillna(df.YearBuilt)

    #Label encoding
    from sklearn.preprocessing import LabelEncoder
    df = df.apply(LabelEncoder().fit_transform)

    data = df.drop(columns=['SalePrice'])
    target = df.SalePrice

    #Splitting and training
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=123)
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.5, learning_rate = 0.1, max_depth = 15, alpha = 10)

    xg_reg.fit(X_train,y_train)


    intermediate = [
        {"Garage":[c for c in df.columns if 'Garage' in c]},
        {"Basement":[c for c in df.columns if 'Bsmt' in c]},
        {"Lot":list(df.columns[[3,4,7,8,9,10,11]])},
        {"Access":list(df.columns[[13,14]])},
        {"House_type":list(df.columns[[1,15,16,21]])},
        {"House_aesthetics":list(df.columns[[22,23,24,25,26]])},
        {"House_condition":list(df.columns[[20,18,21,28,19,29]])},
        {"First_floor_surface":list(df.columns[[43]])},
        {"Above_ground_living area":[c for c in df.columns if 'GrLivArea' in c]}
    ]


    test_data_ames = X_test.iloc[[345]]

    ciu = determine_ciu(
        test_data_ames,
        xg_reg.predict,
        df.to_dict('list'),
        samples = 1000,
        prediction_index = None,
        intermediate_concepts = intermediate
    )

    return ciu, intermediate

