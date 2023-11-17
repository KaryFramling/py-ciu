import numpy as np
import pandas as pd
import random
from .CIUresult import CIUresult

class CIU:
    def __init__(
        self,
        predictor,
        out_names,
        data=None, 
        input_names=None,
        in_minmaxs=None, 
        out_minmaxs=None, 
        nsamples=100,
        category_mapping=None, 
        intermediate_concepts=None, 
        neutralCU=0.5, 
        instance=None
    ):
        """
        @param predictor: Model prediction function to be used.
        @param out_names: Array of names for the model outputs. This parameter is compulsory because
        it is used for determining how many outputs there are and initializing out_minmaxs to 0/1 if 
        they are not provided as parameters.
        @param data: DataFrame with the data set to use for inferring min and max input values. Only 
        needed if in_minmaxs is not provided. Default: ``None``.
        @param input_names: list of input column names in ``data``. Default: ``None``.
        @param in_minmaxs: Pandas DataFrame with columns ``min`` and ``max`` and one row per input. Default: ``None``.
        @param out_minmaxs: Pandas DataFrame with columns ``min`` and ``max`` and one row per 
        model output. Default: ``None``.
        @param instance: Instance to be explained, which can be given directly here, which cause an 
        explanation to be calculated straight away, without requiring a call to explain(). Default: ``None``.
        """
        self.out_names=out_names
        if out_minmaxs is None:
            self.out_minmaxs = pd.DataFrame({'mins': 0, 'maxs': 1}, index=range(len(out_names)))
            self.out_minmaxs.index = out_names
        if in_minmaxs is None:
            try:
                self.in_minmaxs = pd.DataFrame({'mins': data[input_names].min(), 'maxs': data[input_names].max()})
            except:
                print("Logic Error: You must provide either min_max values or a dataset and input names from which they can be inferred.")
                raise
        else: 
            self.in_minmaxs=in_minmaxs
            input_names = list(self.in_minmaxs.index)

        self.predictor=predictor
        self.data=data
        self.input_names=input_names
        self.nsamples=nsamples
        self.category_mapping=category_mapping 
        self.intermediate_concepts=intermediate_concepts 
        self.neutralCU = neutralCU
        self.instance = instance

    def explain_core(self, coalition_inputs, instance=None, nsamples=None, neutralCU = None, 
                     target_inputs=None):
        """
        Calculate CIU for a coalition of inputs. This is the "core" CIU method with the actual 
        CIU calculations. 

        Coalitions of inputs are used for defining CIU's "intermediate concepts". It signifies that all the 
        inputs in the coalition are perturbed at the same time. 
        @param coalition_inputs: array of input indices. 
        """
        if instance is not None:
            self.instance = instance 
        if self.instance is None:
            raise ValueError("No instance to explain has been given.")
        if nsamples is None:
            nsamples = self.nsamples
        if neutralCU is None:
            neutralCU = self.neutralCU
        outvals = self.predictor(self.instance)
        nouts = outvals.shape[1] # Number of outputs.
        samples = self._generate_samples(instance, self.input_names, nsamples, coalition_inputs, {})
        samples_out = self.predictor(samples)
        maxs = np.amax(samples_out,axis=0)
        mins = np.amin(samples_out,axis=0)
        # If "target_inputs" is given, then we need to get "outmin" and "outmax" values for that 
        # coalition of inputs, rather than for the final outputs.
        if target_inputs is not None:
            target_cius = self.explain_core(target_inputs, instance, nsamples, neutralCU)
            # This is not finished yet! 
        else:
            outmins = self.out_minmaxs.iloc[:,0]
            outmaxs = self.out_minmaxs.iloc[:,1]
        cius = []
        for i in range(nouts):
            ci = (maxs[i] - mins[i])/(outmaxs[i] - outmins[i])
            cu = (outvals[0,i] - mins[i])/ci
            cinfl = ci*(cu - neutralCU)
            outname = self.out_names[i]
            fname = self.input_names[coalition_inputs[0]] if len(coalition_inputs) == 1 else "Coalition of %i inputs" % len(coalition_inputs)
            ciu = pd.DataFrame({'ci': [ci], 'cu': [cu], 'cinfl': [cinfl], 'outname': [outname], 
                                'fname': [fname], 'ymin': [mins[i]], 'ymax': [maxs[i]], 
                                'inputs': [coalition_inputs], 'target_inputs': [target_inputs]})
            ciu.index.name = 'Feature'
            ciu.index = [[fname]]
            # ciu = CIUresult(ci, cu, cinfl, mins[i], mins[i], outvals[0,i], fname, outname, instance, self)
            cius.append(ciu)
        return cius

    def explain(self, instance=None, namples=None, neutralCU = None, category_mapping=None, intermediate_concepts=None):
        """
        Determines contextual importance and utility for a given instance (set of input/feature values).
        Or that's what it should do at least, maybe not so for the moment...

        @param instance: Instance to be explained. The default is None.
        @param samples: Number of samples to use. Default is ``None``, which means using the default 
        value of the constructor.
        @param neutralCU: Value to use for "neutral CU" in Contextual influence calculation. 
        Default is ``None`` because this parameter is only intended to temporarily override the value 
        given to the constructor. 
        @param category_mapping: Mapping of one-hot encoded categorical variables to list of 
        categories and category name. Defaults to ``None``.
        @param intermediate_concepts: List of {key: list} tuples of features whose interactions should 
        be evaluated. Defaults to ``[]``.
        """
        # If no instance is given, then we use the one that we had already
        if instance is not None:
            self.instance = instance 
        if self.instance is None:
            raise ValueError("No instance to explain has been given.")
        
        if neutralCU is None:
            neutralCU = self.neutralCU
        
        if nsamples is None:
            nsamples = self.nsamples
        
        prediction_index = 0 # Temporary thing, we return a list with CIUresult for all outputs

        if intermediate_concepts is None:
            intermediate_concepts = []

        if category_mapping is None:
            category_mapping = {}

        for i in self.instance.columns:
            self.instance[i] = self.instance[i].astype(float)

        # This looks a little obscure...?
        category_names = list(category_mapping.keys())
        feature_names_decoded = []
        categories_encoded = []
        for feature in self.input_names:
            is_category = False
            for index, categories in enumerate(category_mapping.values()):
                if feature in categories:
                    categories_encoded.append(feature)
                    is_category = True
                    if category_names[index] not in feature_names_decoded:
                        feature_names_decoded.append(category_names[index])
            if not is_category:
                feature_names_decoded.append(feature)

        cis = [0.1,0.2,0.3]
        cus = [0.4,0.5,0.6]
        cinfls = [1,2,3]
        c_mins = [0,0,0]
        c_maxs = [1,1,1]
        outvals = self.predictor(self.instance)

        all_samples = []

        # Now this actually systematically calculates CIU for all inputs.
        for i, feature_i in enumerate(self.input_names):
            feature_samples = self._generate_samples(self.instance, self.input_names, nsamples, [i], category_mapping)
            all_samples.append(feature_samples)

        joined_samples = pd.concat(all_samples)
        #joined_samples = [j for i in all_samples for j in i]
        print(joined_samples.shape)
 
        # try:
        #     all_samples_flat = pd.DataFrame(joined_samples)
        #     all_preds = predictor(all_samples_flat)
        # except ValueError:
        #     all_samples_flat = pd.DataFrame(joined_samples, dtype=float)
        #     all_preds = predictor(all_samples_flat)

        # predictions = {feat: all_preds[ind * samples + ind: (ind + 1) * samples + ind + 1] if prediction_index is None else \
        #     [prob[prediction_index] for prob in all_preds[ind * samples + ind: (ind + 1) * samples + ind + 1]] \
        #             for ind, feat in enumerate(in_minmaxs.keys())}

        #for intermediate_concept in intermediate_concepts:
        #    interaction_name = list(intermediate_concept.keys())[0]
        #    features = list(intermediate_concept.values())[0]
        #    indices = [list(in_minmaxs.keys()).index(feature) for feature in features]

        #    feature_samples = _generate_samples(
        #        case.iloc[0, :].to_dict(), in_minmaxs.keys(), in_minmaxs, samples, indices, category_mapping
        #    )

        #    try:
        #        feature_samples = pd.DataFrame(feature_samples)
        #        predictions[interaction_name] = \
        #            predictor(feature_samples) if prediction_index is None \
        #                else [prob[prediction_index] for \
        #                    prob in predictor(feature_samples)]
        #    except ValueError:
        #        feature_samples = pd.DataFrame(feature_samples, dtype=float)
        #        predictions[interaction_name] = \
        #            predictor(feature_samples) if prediction_index is None \
        #                else [prob[prediction_index] for \
        #                    prob in predictor(feature_samples)]

        abs_max = None
        abs_min = None

        # determine absolute min/max, only considering single features
        #for feature in feature_names_decoded:
            # Get right predictions for decoded category
        #    if feature in category_mapping.keys():
        #        encoded_feature = None
        #        for encoded_feature_j in in_minmaxs.keys():
        #            feature_max = max(predictions[encoded_feature_j])
        #            if abs_max is None or abs_max < feature_max:
        #                abs_max = feature_max
        #            feature_min = min(predictions[encoded_feature_j])
        #            if abs_min is None or abs_min > feature_min:
        #                abs_min = feature_min
        #    else:
        #        feature_max = max(predictions[feature])
        #        if abs_max is None or abs_max < feature_max:
        #            abs_max = feature_max

        #        feature_min = min(predictions[feature])
        #        if abs_min is None or abs_min > feature_min:
        #            abs_min = feature_min

        # determine absolute min/max, also considering feature interactions
        #for intermediate_concept in intermediate_concepts:
        #    intermediate_concept_name = list(intermediate_concept.keys())[0]
        #    interaction_max = max(predictions[intermediate_concept_name])
        #    if abs_max is None or abs_max < interaction_max:
        #        abs_max = interaction_max

        #    interaction_min = min(predictions[intermediate_concept_name])
        #    if abs_min is None or abs_min > interaction_min:
        #        abs_min = interaction_min

        # compute CI/CU for single features
        #for index, feature in enumerate(feature_names_decoded):
            # Get right predictions for decoded category
        #    if feature in category_mapping.keys():
        #        encoded_feature = None
        #        for encoded_feature_j in in_minmaxs.keys():
        #            if encoded_feature_j in category_mapping[feature] and list(case[encoded_feature_j].to_dict().values())[0] == 1:
        #                encoded_feature = encoded_feature_j
        #        c_min = min(predictions[encoded_feature])
        #        c_max = max(predictions[encoded_feature])
        #    else:
        #        c_min = min(predictions[feature])
        #        c_max = max(predictions[feature])

        #    n = case_prediction
        #    ci = (c_max - c_min) / (abs_max - abs_min)
        #    if (c_max - c_min) == 0:
        #        cu = (n - c_min) / 0.01
        #    else:
        #        cu = (n - c_min) / (c_max - c_min)
        #    if cu == 0: cu = 0.001
        #    cis[feature] = ci
        #    cus[feature] = cu
        #    c_mins[feature] = c_min
        #    c_maxs[feature] = c_max

        # compute CI/CU for feature interactions
        intermediate_concept_names = [
            "_".join(features) for features in intermediate_concepts
        ]

        #for intermediate_concept_name in intermediate_concept_names:
        #    c_min = min(predictions[intermediate_concept_name])
        #    c_max = max(predictions[intermediate_concept_name])
        #    n = case_prediction
        #    ci = (c_max - c_min) / (abs_max - abs_min)
        #    if (c_max - c_min) == 0:
        #        cu = (n - c_min) / 0.01
        #    else:
        #        cu = (n - c_min) / (c_max - c_min)
        #    if cu == 0: cu = 0.001
        #    cis[intermediate_concept_name] = ci
        #    cus[intermediate_concept_name] = cu
        #    c_mins[intermediate_concept_name] = c_min
        #    c_maxs[intermediate_concept_name] = c_max

        # Return a list of results, one for each output. It there's only one output, then 
        # it will be a list with only one element. 
        #return CIUresult(cis, cus, cinfls, c_mins, c_maxs, outvals, intermediate_concepts, intermediate_concept_names, instance, self)
        return CIUresult(cis, cus, cinfls, c_mins, c_maxs, outvals[0][2], intermediate_concepts, ['titi','toto','tata'], instance, self)

    def _generate_samples(self, instance, feature_names, samples, indices, category_mapping):
        """
        Generate a list of instances for estimating CIU.

        @param instance: The instance to generate the permuted instances for.
        @param feature_names: Blabla.
        @return: DataFrame with perturbed instances.
        """
        rows = [instance]
        # Here should be the adding of the min/max values of the input indices to modify!
        for _ in range(samples):
            sample_entry = instance.copy()
            for index_j, feature_j in enumerate(feature_names):
                if index_j in indices:
                    min_val = self.in_minmaxs.loc[feature_j][0]
                    max_val = self.in_minmaxs.loc[feature_j][1]
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

        return pd.concat(rows)
     
