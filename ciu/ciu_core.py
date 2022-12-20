import pandas as pd
pd.options.mode.chained_assignment = None
import random
from ciu.ciu_object import CiuObject
import numpy as np


def _generate_samples(case, feature_names, min_maxs, samples, indices,
                      category_mapping):
    rows = [case]
    for _ in range(samples):
        sample_entry = {}
        for index_j, feature_j in enumerate(feature_names):
            if index_j not in indices:
                # if not (index_j in indices):
                sample_entry[feature_j] = case[feature_j]
            else:
                min_val = min_maxs[feature_j][0]
                max_val = min_maxs[feature_j][1]
                sample_entry[feature_j] = random.uniform(min_val, max_val)
                # check if feature_j, feature_k in same category;
                # if so, set feature_k to 0 if feature_j is 1
                for index_k, feature_k in enumerate(feature_names):
                    if index_j != index_k:
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
                try:
                    if (sample_entry[category] == 1).any(): is_activated = True
                except AttributeError:
                    if sample_entry[category] == 1: is_activated = True
            if not is_activated:
                category = categories[random.randint(0, len(categories) - 1)]
                sample_entry[category] = 1
        rows.append(sample_entry)

    return rows


def determine_ciu(
        case, predictor, dataset=None, min_maxs=None, samples=1000,
        prediction_index=None, category_mapping=None, intermediate_concepts=None):
    """
    Determines contextual importance and utility for a given case.

    :param case: Case data (Pandas DataFrame)
    :param predictor: The prediction function of the black-box model Py-CIU should call
    :param dict dataset: Dataset to deduct min_maxs from (dictionary).
                            Defaults to ``None``.
    :param min_maxs: dictionary (``'feature_name': [min, max, is_int]`` for each feature),
                    or infered from dataset. Defaults to ``None``
    :param samples: number of samples to be generated. Defaults to 1000.
    :param prediction_index: If the model returns several predictions, provide
                             the index of the relevant prediction. Defaults to
                             ``None``
    :param category_mapping: Mapping of one-hot encoded categorical variables to
                             list of categories and category name. Defaults to
                             ``None``.
    :param intermediate_concepts: List of {key: list} tuples of features whose
                                 interactions should be evaluated. Defaults to
                                 ``[]``.

    :return: dictionary: for each feature: list with
             contextual importance value, contextual utility value
    """

    if intermediate_concepts is None:
        intermediate_concepts = []

    if category_mapping is None:
        category_mapping = {}

    for i in case.columns:
        case[i] = case[i].astype(float)

    if min_maxs is None:
        try:
            min_maxs = {i: [min(list(map(float, dataset[i]))), max(list(map(float, dataset[i]))),
                            isinstance(min(dataset[i]), float)] for i in case.columns}
        except:
            print("Logic Error: You must provide either min_max values or a dataset to infer them from")
            raise

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
    c_mins = {}
    c_maxs = {}
    outval = {prediction_index: predictor(case)[0][prediction_index] if prediction_index is not None else predictor(case)[0]}

    case_prediction = \
        predictor(case)[0] if prediction_index is None \
            else predictor(case)[0][prediction_index]

    all_samples = []

    for index_i, feature_i in enumerate(min_maxs.keys()):
        feature_samples = _generate_samples(
            case.iloc[0, :].to_dict(), min_maxs.keys(), min_maxs, samples, [index_i],
            category_mapping
        )
        all_samples.append(feature_samples)

    joined_samples = [j for i in all_samples for j in i]

    try:
        all_samples_flat = pd.DataFrame(joined_samples)
        all_preds = predictor(all_samples_flat)
    except ValueError:
        all_samples_flat = pd.DataFrame(joined_samples, dtype=float)
        all_preds = predictor(all_samples_flat)

    predictions = {feat: all_preds[ind * samples + ind: (ind + 1) * samples + ind + 1] if prediction_index is None else \
        [prob[prediction_index] for prob in all_preds[ind * samples + ind: (ind + 1) * samples + ind + 1]] \
                   for ind, feat in enumerate(min_maxs.keys())}

    for intermediate_concept in intermediate_concepts:
        interaction_name = list(intermediate_concept.keys())[0]
        features = list(intermediate_concept.values())[0]
        indices = [list(min_maxs.keys()).index(feature) for feature in features]

        feature_samples = _generate_samples(
            case.iloc[0, :].to_dict(), min_maxs.keys(), min_maxs, samples, indices, category_mapping
        )

        try:
            feature_samples = pd.DataFrame(feature_samples)
            predictions[interaction_name] = \
                predictor(feature_samples) if prediction_index is None \
                    else [prob[prediction_index] for \
                          prob in predictor(feature_samples)]
        except ValueError:
            feature_samples = pd.DataFrame(feature_samples, dtype=float)
            predictions[interaction_name] = \
                predictor(feature_samples) if prediction_index is None \
                    else [prob[prediction_index] for \
                          prob in predictor(feature_samples)]

    abs_max = None
    abs_min = None

    # determine absolute min/max, only considering single features
    for feature in feature_names_decoded:
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
    for intermediate_concept in intermediate_concepts:
        intermediate_concept_name = list(intermediate_concept.keys())[0]
        interaction_max = max(predictions[intermediate_concept_name])
        if abs_max is None or abs_max < interaction_max:
            abs_max = interaction_max

        interaction_min = min(predictions[intermediate_concept_name])
        if abs_min is None or abs_min > interaction_min:
            abs_min = interaction_min

    # compute CI/CU for single features
    for index, feature in enumerate(feature_names_decoded):
        # Get right predictions for decoded category
        if feature in category_mapping.keys():
            encoded_feature = None
            for encoded_feature_j in min_maxs.keys():
                if encoded_feature_j in category_mapping[feature] and list(case[encoded_feature_j].to_dict().values())[0] == 1:
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
        c_mins[feature] = c_min
        c_maxs[feature] = c_max

    # compute CI/CU for feature interactions
    intermediate_concept_names = [
        "_".join(features) for features in intermediate_concepts
    ]

    for intermediate_concept_name in intermediate_concept_names:
        c_min = min(predictions[intermediate_concept_name])
        c_max = max(predictions[intermediate_concept_name])
        n = case_prediction
        ci = (c_max - c_min) / (abs_max - abs_min)
        if (c_max - c_min) == 0:
            cu = (n - c_min) / 0.01
        else:
            cu = (n - c_min) / (c_max - c_min)
        if cu == 0: cu = 0.001
        cis[intermediate_concept_name] = ci
        cus[intermediate_concept_name] = cu
        c_mins[intermediate_concept_name] = c_min
        c_maxs[intermediate_concept_name] = c_max

    return CiuObject(cis, cus, c_mins, c_maxs, outval, intermediate_concepts, intermediate_concept_names, case, predictor, min_maxs)