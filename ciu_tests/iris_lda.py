
def get_iris_test():
    """
    :return: iris LDA CIU Object, LDA model, example instance
    """

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn import datasets
    import ciu as ciu
    from ciu.CIU import CIU


    iris=datasets.load_iris()

    df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                  columns = iris['feature_names'] + ['target'])
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']
    iris_outnames = df['species'].cat.categories.tolist()
    X = df[['s_length', 's_width', 'p_length', 'p_width']]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    #iris_df = df.apply(pd.to_numeric, errors='ignore')

    #Can also be written manually:
    test_iris = pd.DataFrame.from_dict({'s_length' : [2.0], 's_width' : [3.2], 'p_length': [1.8], 'p_width' : [2.4]})

    ciu = CIU(model.predict_proba, iris_outnames, data=X_train)

    return ciu, model, test_iris