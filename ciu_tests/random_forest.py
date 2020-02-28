from sklearn.ensemble import RandomForestClassifier
from ciu import determine_ciu

from loan_data_generator import generate_data


def generate_model(train_data):
    """
    :param train_data: dataframe to train on
    Generates example model to test CIU
    :return: example model
    """
    labelled_data = train_data
    random_forest = RandomForestClassifier(
        n_estimators=1000,
        random_state=42
    )

    labels = labelled_data[['approved']].values.ravel()
    data = labelled_data.drop(['approved'], axis=1)
    random_forest.fit(data, labels)
    return random_forest

