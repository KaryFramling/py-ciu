import random
import pandas as pd

from ciu.ciu_object import CiuObject


def _generate_samples(case, feature_names, min_maxs, samples, indices,
                      category_mapping):
    rows = []
    rows.append(case)
    for sample in range(samples):
        sample_entry = {}
        for index_j, feature_j in enumerate(feature_names):
            if not (index_j in indices):
                # if not (index_j in indices):
                sample_entry[feature_j] = case[feature_j]
            else:
                min_val = min_maxs[feature_j][0]
                max_val = min_maxs[feature_j][1]
                is_int = min_maxs[feature_j][2]
                sample_entry[feature_j] = \
                    random.randint(min_val, max_val) if is_int \
                        else random.uniform(min_val, max_val)
                # check if feature_j, feature_k in same category;
                # if so, set feature_k to 0 if feature_j is 1
                for index_k, feature_k in enumerate(feature_names):
                    if not (index_j == index_k):
                        for categories in category_mapping.values():
                            same_category = feature_j in categories \
                                            and feature_k in categories
                            if same_category and sample_entry[feature_j] == 1:
                                sample_entry[feature_k] = 0
        # set categorical values that would otherwise not have a category
        # assigned
        for categories in category_mapping.values():
            is_activated = False
            for category in categories:
                if sample_entry[category] == 1: is_activated = True
            if not is_activated:
                category = categories[random.randint(0, len(categories) -1)]
                sample_entry[category] = 1
        rows.append(sample_entry)
    return pd.DataFrame(rows)


def determine_ciu(
        case, predictor, min_maxs, samples=1000,
        prediction_index=None, category_mapping=None, feature_interactions=[]):
    """
    Determines contextual importance and utility for a given case.

    :param case: Case data (dictionary)
    :param predictor: The prediction function of the black-box model Py-CIU should call
    :param min_maxs: dictionary (``'feature_name': [min, max, is_int]`` for each feature)
    :param samples: number of samples to be generated. Defaults to 1000.
    :param prediction_index: If the model returns several predictions, provide
                             the index of the relevant prediction. Defaults to
                             ``None``
    :param category_mapping: Mapping of one-hot encoded categorical variables to
                             list of categories and category name. Defaults to
                             ``None``.
    :param feature_interactions: List of {key: list} tuples of features whose
                                 interactions should be evaluated. Defaults to
                                 ``[]``.

    :return: dictionary: for each feature: list with
             contextual importance value, contextual utility value
    """

    if category_mapping is None:
        category_mapping = {}

    category_names = list(category_mapping.keys())
    feature_names_decoded = []
    categories_encoded = []
    for feature in min_maxs.keys():
        is_category = False
        for index, categories in enumerate(category_mapping.values()):
            if feature in categories:
                categories_encoded.append(feature)
                is_category = True
                if category_names[index] not in feature_names_decoded:
                    feature_names_decoded.append(category_names[index])
        if not is_category:
            feature_names_decoded.append(feature)

    cis = {}
    cus = {}

    case_prediction = \
        predictor([list(case.values())])[0] if prediction_index is None \
        else predictor([list(case.values())])[0][prediction_index]

    predictions = {}

    for index_i, feature_i in enumerate(min_maxs.keys()):
        feature_samples = _generate_samples(
            case, min_maxs.keys(), min_maxs, samples, [index_i],
            category_mapping
        )
        predictions[feature_i] = \
            predictor(feature_samples) if prediction_index is None \
            else [prob[prediction_index] for \
                  prob in predictor(feature_samples)]

    for feature_interaction in feature_interactions:
        interaction_name = list(feature_interaction.keys())[0]
        features = list(feature_interaction.values())[0]
        indices = [list(min_maxs.keys()).index(feature) for feature in features]
        feature_samples = _generate_samples(
            case, min_maxs.keys(), min_maxs, samples, indices, category_mapping
        )
        predictions[interaction_name] = \
            predictor(feature_samples) if prediction_index is None \
                else [prob[prediction_index] for \
                      prob in predictor(feature_samples)]
    abs_max = None
    abs_min = None

    # determine absolute min/max, only considering single features
    for index, feature in enumerate(feature_names_decoded):
        # Get right predictions for decoded category
        if feature in category_mapping.keys():
            encoded_feature = None
            for encoded_feature_j in min_maxs.keys():
                feature_max = max(predictions[encoded_feature_j])
                if abs_max is None or abs_max < feature_max:
                    abs_max = feature_max
                feature_min = min(predictions[encoded_feature_j])
                if abs_min is None or abs_min > feature_min:
                    abs_min = feature_min
        else:
            feature_max = max(predictions[feature])
            if abs_max is None or abs_max < feature_max:
                abs_max = feature_max

            feature_min = min(predictions[feature])
            if abs_min is None or abs_min > feature_min:
                abs_min = feature_min

    # determine absolute min/max, also considering feature interactions
    for feature_interaction in feature_interactions:
        interaction_name = list(feature_interaction.keys())[0]
        interaction_max = max(predictions[interaction_name])
        if abs_max is None or abs_max < interaction_max:
            abs_max = interaction_max

        interaction_min = min(predictions[interaction_name])
        if abs_min is None or abs_min > interaction_min:
            abs_min = interaction_min

    # compute CI/CU for single features
    for index, feature in enumerate(feature_names_decoded):
        # Get right predictions for decoded category
        if feature in category_mapping.keys():
            encoded_feature = None
            for encoded_feature_j in min_maxs.keys():
                if case[encoded_feature_j] == 1 and encoded_feature_j in category_mapping[feature]:
                    encoded_feature = encoded_feature_j
            c_min = min(predictions[encoded_feature])
            c_max = max(predictions[encoded_feature])
        else:
            c_min = min(predictions[feature])
            c_max = max(predictions[feature])
        n = case_prediction
        ci = (c_max - c_min) / (abs_max - abs_min)
        if (c_max - c_min) == 0:
            cu = (n - c_min) / 0.01
        else:
            cu = (n - c_min) / (c_max - c_min)
        if cu == 0: cu = 0.001
        cis[feature] = ci
        cus[feature] = cu

    # compute CI/CU for feature interactions
    interaction_names = [
        "_".join(features) for features in feature_interactions
    ]
    for interaction_name in interaction_names:
        c_min = min(predictions[interaction_name])
        c_max = max(predictions[interaction_name])
        n = case_prediction
        ci = (c_max - c_min) / (abs_max - abs_min)
        if (c_max - c_min) == 0:
            cu = (n - c_min) / 0.01
        else:
            cu = (n - c_min) / (c_max - c_min)
        if cu == 0: cu = 0.001
        cis[interaction_name] = ci
        cus[interaction_name] = cu

    ciu = CiuObject(cis, cus, interaction_names)

    return ciu
