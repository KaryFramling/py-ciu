
def get_iris_test():
    """
    :return: iris LDA CIU Object
    """

    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    from ciu.ciu_core import determine_ciu

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