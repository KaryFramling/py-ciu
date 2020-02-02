

def determine_ciu(
        case, model, sample_dists, feature_names, prediction_index=None,
        category_mapping=None):
    """
    Determines contextual importance and utility for a given case.

    :param case: Case data
    :param model: Black-box model that predicts the case outcome
    :param sample_dists: pandas data frame with real or simulated example
                         distributions of all features
    :param feature_names: list of feature names
    :param prediction_index: If the model returns several predictions, provide
                             the index of the relevant prediction. Defaults to
                             ``None``
    :param category_mapping: Mapping of one-hot encoded categorical variables to
                             list of categories and category name. Defaults to
                             ``None``.

    :return: dictionary: for each feature: list with
             contextual importance value, contextual utility value
    """

    if category_mapping is None:
        category_mapping = {}

    category_names = list(category_mapping.keys())
    feature_names_decoded = []
    categories_encoded = []
    for feature in feature_names:
        is_category = False
        for index, categories in enumerate(category_mapping.values()):
            if feature in categories:
                categories_encoded.append(feature)
                is_category = True
                if category_names[index] not in feature_names_decoded:
                    feature_names_decoded.append(category_names[index])
        if not is_category:
            feature_names_decoded.append(feature)

    result = {}
    case_prediction = \
        model.predict_proba([case])[0] if prediction_index is None \
        else model.predict_proba([case])[0][prediction_index]

    predictions = {}

    for index_i, feature_i in enumerate(feature_names):
        feature_samples = sample_dists.copy()
        for index_j, feature_j in enumerate(feature_names):
            if not (index_j == index_i):
                feature_samples[feature_j] = case[index_j]
                # check if feature_i, feature_i in same category;
                # if so, set all but feature_i to negation of feature_i
                for categories in category_mapping.values():
                    same_category = feature_j in categories \
                                    and feature_i in categories
                    if same_category and case[index_j] == 1:
                            feature_samples[feature_i] = 0
        predictions[feature_i] = \
            model.predict_proba(feature_samples) if prediction_index is None \
            else [prob[prediction_index] for \
                  prob in model.predict_proba(feature_samples)]

    abs_max = None
    abs_min = None

    for index, feature in enumerate(feature_names_decoded):
        # Get right predictions for decoded category
        if feature in category_mapping.keys():
            encoded_feature = None
            for index_j, encoded_feature_j in enumerate(feature_names):
                if case[index_j] == 1:
                    encoded_feature = encoded_feature_j
            feature_max = max(predictions[encoded_feature])
            if abs_max is None or abs_max < feature_max:
                abs_max = feature_max

            feature_min = min(predictions[encoded_feature])
            if abs_min is None or abs_min > feature_min:
                abs_min = feature_min
        else:
            feature_max = max(predictions[feature])
            if abs_max is None or abs_max < feature_max:
                abs_max = feature_max

            feature_min = min(predictions[feature])
            if abs_min is None or abs_min > feature_min:
                abs_min = feature_min

    print(f'abs_max: {abs_max}, abs_min: {abs_min}')
    for index, feature in enumerate(feature_names_decoded):
        # Get right predictions for decoded category
        if feature in category_mapping.keys():
            encoded_feature = None
            for index_j, encoded_feature_j in enumerate(feature_names):
                if case[index_j] == 1:
                    encoded_feature = encoded_feature_j
            c_min = min(predictions[encoded_feature])
            c_max = max(predictions[encoded_feature])
        else:
            c_min = min(predictions[feature])
            c_max = max(predictions[feature])
        print(f'c_max: {c_max}, c_min: {c_min}')
        n = case_prediction
        print(f'n: {n}')
        ci = (c_max - c_min) / (abs_max - abs_min)
        cu = (n - c_min) / (c_max - c_min)
        print(f'ci: {ci}, cu: {cu}')
        result[feature] = [ci, cu]

    return result
