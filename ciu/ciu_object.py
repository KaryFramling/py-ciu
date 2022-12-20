from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class CiuObject:
    def __init__(self, ci, cu, c_mins, c_maxs, outval, intermediate_concepts, concept_names, case, predictor, min_maxs):
        self.ci = ci
        self.cu = cu
        self.c_mins = c_mins
        self.c_maxs = c_maxs
        self.outval = outval
        self.intermediate_concepts = intermediate_concepts
        self.concept_names = concept_names
        self.case = case
        self.predictor = predictor
        self.min_maxs = min_maxs

    def _get_target_concept(self, target_concept, ind_inputs=None):
        # Initialising an ideal inputs copy including all the inputs of the concept
        ind_inputs_copy = []
        out_ci = self.ci.copy()
        for concept in self.intermediate_concepts:
            if target_concept in concept.keys():
                for feature_list in concept.values():
                    for feature in feature_list:
                        out_ci[feature] = (self.c_maxs[feature] - self.c_mins[feature]) / (
                                    self.c_maxs[target_concept] - self.c_mins[target_concept])
                        ind_inputs_copy.append(list(self.ci.keys()).index(feature))

        # Checking if there are user inputs, otherwise setting them to ideal copy
        if ind_inputs is None:
            ind_inputs = ind_inputs_copy

        # If there are inputs but outside of concept, toss a warning and set to ideal copy
        elif len(ind_inputs) >= 1:
            for i in ind_inputs:
                if i not in ind_inputs_copy:
                    print("WARNING: The indices selected must be a subset of the target concept.\n"
                          f"Index number {i} is outside of the concept scope.")
                    ind_inputs = ind_inputs_copy
                    break

        return out_ci, ind_inputs

    @staticmethod
    def _filter_feature_names(feature_names, concept_names, include_intermediate_concepts, ind_inputs=None):
        feature_names_final = []
        original = [x for x in feature_names if x not in concept_names]
        exclude_intermediate_concepts = include_intermediate_concepts == 'no'
        exclude_ordinary = include_intermediate_concepts == 'only'
        for feature_name in feature_names:
            exclude_feature = \
                (exclude_intermediate_concepts and feature_name in concept_names) or \
                (exclude_ordinary and feature_name not in concept_names)
            if not exclude_feature:
                if exclude_ordinary:
                    feature_names_final.append(feature_name)
                if not exclude_ordinary and ind_inputs is None:
                    feature_names_final.append(feature_name)
            if ind_inputs:
                feature_names_final.extend(
                    list(original)[index] for index in ind_inputs if list(original)[index] not in feature_names_final)
        return feature_names_final

    def explain_tabular(self, include_intermediate_concepts=None, ind_inputs=None, target_concept=None):
        """
        :param str target_concept: defines which intermediate concept to explain;
        :param list ind_inputs: list of feature indexes to produce a tabular explanation for, it will include all of them by default;
                                NOTE: this can add extra indexes to explain even if the include_intermediate_concepts param is set to 'only'
        :param str include_intermediate_concepts: define whether to include 'no' intermediate concepts or 'only' intermediate concepts;
                                                    it will include all intermediate concepts and all independent features by default
        :return: dataframe output_df: Pandas dataframe containing the output data
        """
        if target_concept:
            # Removing the inclusion of the other concepts automatically
            include_intermediate_concepts = None
            out_ci, ind_inputs = self._get_target_concept(
                target_concept, ind_inputs
            )

        if target_concept is None:
            out_ci = self.ci

        out_df = pd.DataFrame.from_dict([out_ci,
                                         self.cu,
                                         self.c_mins,
                                         self.c_maxs])

        out_df.index = pd.Index(['CI', 'CU', 'cmin', 'cmax'], name='Features')

        feature_names_out = self._filter_feature_names(
            self.ci.keys(),
            self.concept_names,
            include_intermediate_concepts,
            ind_inputs
        )

        out_df = out_df[feature_names_out]

        output_df = out_df.T
        output_df['outval'] = list(self.outval.values())[0]

        return output_df

    def explain_text(self, include_intermediate_concepts=None, ind_inputs=None, thresholds_ci=None, thresholds_cu=None, target_concept=None):
        """
        :param str target_concept: defines which intermediate concept to explain;
        :param list ind_inputs: list of feature indexes to produce a textual explanation for, it will include all of them by default;
                                NOTE: this can add extra indexes to explain even if the include_intermediate_concepts param is set to 'only'
        :param str include_intermediate_concepts: define whether to include 'no' intermediate concepts or 'only' intermediate concepts;
                                                    it will include all intermediate concepts and all independent features by default
        :param dict thresholds_ci: dictionary containing the label and ceiling value for the CI thresholds
        :param dict thresholds_cu: dictionary containing the label and ceiling value for the CU thresholds
        :return: list explanation_texts: list containing explanation strings
        """

        if thresholds_ci is None:
            thresholds_ci = {
                'very low importance': 0.20,
                'low importance': 0.40,
                'normal importance': 0.60,
                'high importance': 0.80,
                'very high importance': 1
            }

        if thresholds_cu is None:
            thresholds_cu = {
                'not typical': 0.25,
                'somewhat typical': 0.5,
                'typical': 0.75,
                'very typical': 1
            }

        if len(thresholds_cu) < 2 or len(thresholds_ci) < 2:
            raise ValueError(f"The dictionaries containing the CI/CU thresholds must have at least 2 elements. \
                             \nCI dict: {thresholds_ci} \nCU dict: {thresholds_cu}")

        if target_concept:
            #Removing the inclusion of the other concepts automatically
            include_intermediate_concepts = None
            out_ci, ind_inputs = self._get_target_concept(
                target_concept, ind_inputs
            )

        if target_concept is None:
            out_ci = self.ci

        feature_names = self._filter_feature_names(
            self.ci.keys(),
            self.concept_names,
            include_intermediate_concepts,
            ind_inputs
        )

        explanation_texts = []

        if target_concept:
            for k, v in thresholds_cu.items():
                if self.cu[target_concept] <= v:
                    cu_concept_text = k
                    break

            cu_concept = round(self.cu[target_concept] * 100, 2)

            concept_text = f'The intermediate concept "{target_concept}", is {cu_concept_text} ' \
                                       f'for its prediction (CU={cu_concept}%).'
            explanation_texts.append(concept_text)

        for feature in list(feature_names):
            try:
                for k, v in thresholds_ci.items():
                    if out_ci[feature] <= v:
                        ci_text = k
                        break

                for k, v in thresholds_cu.items():
                    if self.cu[feature] <= v:
                        cu_text = k
                        break
            except TypeError as e:
                raise TypeError(f"The dictionaries containing the CI/CU thresholds cannot have \x1B[3mNone\x1B[0m values. \
                                    \nCI dict: {thresholds_ci} \nCU dict: {thresholds_cu}") from e


            ci = round(out_ci[feature] * 100, 2)
            cu = round(self.cu[feature] * 100, 2)

            explanation_text = f'The feature "{feature}", which is of ' \
                                       f'{ci_text} (CI={ci}%), is {cu_text} ' \
                                       f'for its prediction (CU={cu}%).'

            explanation_texts.append(explanation_text)

        return explanation_texts

    def plot_3D(self, ind_inputs=None):
        """
        :param list ind_inputs: indexes for two features to use for the 3D plot explanation.
        :return: 3D plot object
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))

        # Grabbing data
        data = np.fromiter(self.ci.values(), dtype=float)
        min_maxs = self.min_maxs

        feature_names_prelim = self.ci.keys()
        feature_names = self._filter_feature_names(
            feature_names_prelim,
            self.concept_names,
            include_intermediate_concepts = None,
            ind_inputs = ind_inputs
        )

        # Removing redundant ones
        indices_deleted = 0
        for index, feature_name in enumerate(feature_names_prelim):
            if feature_name not in feature_names:
                data = np.delete(data, index - indices_deleted)
                indices_deleted += 1

        # Making function to simulate R expand_grid iterative functionality
        from itertools import product

        def expand_grid(dictionary):
            return pd.DataFrame(list(product(*dictionary.values())), columns=dictionary.keys())


        xp = np.arange(min_maxs[feature_names[0]][0], min_maxs[feature_names[0]][1],
                       (min_maxs[feature_names[0]][1] - min_maxs[feature_names[0]][0])/40)
        yp = np.arange(min_maxs[feature_names[1]][0], min_maxs[feature_names[1]][1],
                       (min_maxs[feature_names[1]][1] - min_maxs[feature_names[1]][0])/40)
        pm = expand_grid({feature_names[0]: xp,
                           feature_names[1]: yp})

        # PD does not duplicate values on concat, rather sets them to NaN, so repeating manually
        m = pd.DataFrame(np.repeat(self.case.values, len(pm.index), axis=0))
        m.columns = self.case.columns

        # Replacing default values
        m.update(pm)

        z = self.predictor(m)

        # Extracting index we want and reshaping to square matrix
        index = list(self.outval.keys())[0]
        index_out = int(index) if index else 0

        # Checking for single output scenarios
        try:
            zm = np.reshape(z[:,index_out], (len(xp), len(xp)))
        except IndexError:
            zm = np.reshape(z, (len(xp), len(xp)))

        xi, yi = np.meshgrid(xp, yp)

        ax.plot_surface(xi, yi, zm, color="lightblue", linewidth=1, antialiased=True, zorder=1, alpha=0.7)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])

        # Changing azimuth slightly, looks better
        ax.azim += 10

        # Adding instance point marker
        ax.scatter(self.case[feature_names[0]], self.case[feature_names[1]], list(self.outval.values())[0], color="red", alpha=1, s=100, zorder=3)
        fig.suptitle(f"Prediction Index {index_out} ({list(self.outval.values())[0]})")


    def plot_ciu(self, plot_mode='default', include_intermediate_concepts=None, use_influence=False,
                 ind_inputs=None, target_concept=None, sort='ci', color_blind=None,
                 color_fill_ci='#7fffd44d', color_edge_ci='#66CDAA',
                 color_fill_cu="#006400cc", color_edge_cu="#006400"):

        """
        :param bool use_influence: if True the plot will use Contextual Influence;
        :param str target_concept: defines which intermediate concept to explain;
        :param list ind_inputs: list of feature indexes to produce a plot explanation for, it will include all of them by default;
                                NOTE: this can add extra indexes to explain even if the include_intermediate_concepts param is set to 'only'
        :param str include_intermediate_concepts: define whether to include 'no' intermediate concepts or 'only' intermediate concepts;
                                                    otherwise it will include all intermediate concepts and all independent features by default
        :param str plot_mode: defines the type plot to use between 'default', 'overlap' and 'combined'.
        :param str sort: defines the order of the plot bars by the 'ci' (default), 'cu' values or unsorted if None;
        :param str color_blind: defines accessible color maps to use for the plots, such as 'protanopia',
                                                        'deuteranopia' and 'tritanopia'.
        :param str color_edge_cu: defines the hex or named color for the CU edge in the overlap plot mode.
        :param str color_fill_cu: defines the hex or named color for the CU fill in the overlap plot mode.
        :param str color_edge_ci: defines the hex or named color for the CI edge in the overlap plot mode.
        :param str color_fill_ci: defines the hex or named color for the CI fill in the overlap plot mode.
        """

        if target_concept:
            #Removing the inclusion of the other concepts automatically
            include_intermediate_concepts = None
            out_ci, ind_inputs = self._get_target_concept(
                target_concept, ind_inputs
            )

        if target_concept is None:
            out_ci = self.ci

        influence = {}
        if use_influence:
            for k, v in out_ci.items():
                influence[k] = v*(self.cu[k]-0.5)
            out_ci = influence

        data = np.fromiter(out_ci.values(), dtype=float)
        cu = np.fromiter(self.cu.values(), dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6))

        if target_concept:
            fig.suptitle(f"The target concept is {target_concept}")

        feature_names_prelim = self.ci.keys()
        feature_names = self._filter_feature_names(
            feature_names_prelim,
            self.concept_names,
            include_intermediate_concepts,
            ind_inputs
        )

        indices_deleted = 0
        for index, feature_name in enumerate(feature_names_prelim):
            if feature_name not in feature_names:
                data = np.delete(data, index - indices_deleted)
                cu = np.delete(cu, index - indices_deleted)
                indices_deleted += 1


        y_pos = np.arange(len(feature_names))

        if sort in ['ci', 'influence']:
            data, cu, feature_names = (list(t) for t in zip(*sorted(zip(data, cu, feature_names))))
        elif sort == 'cu':
            cu, data, feature_names = (list(t) for t in zip(*sorted(zip(cu, data, feature_names))))

        my_norm = colors.Normalize(vmin=0, vmax=1)
        nodes = [0.0, 0.5, 1.0]

        if color_blind is None:
            colours = ["red", "yellow", "green"]
        elif color_blind == 'protanopia':
            colours = ["gray", "yellow", "blue"]
        elif color_blind == 'deuteranopia':
            colours = ["slategray", "orange", "dodgerblue"]
        elif color_blind == 'tritanopia':
            colours = ["#ff0066", "#ffe6f2", "#00e6e6"]

        cmap1 = colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colours)))


        sm = cm.ScalarMappable(cmap=cmap1, norm=my_norm)
        sm.set_array([])

        if plot_mode == "default":
            if use_influence:
                plt.xlabel("ϕ")

                for m in range(len(data)):
                  ax.barh(y_pos[m], data[m], color=["red" if data[m] < 0 else "blue"],
                          edgecolor="#808080", zorder=2)
            else:
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_label('CU', rotation=0, labelpad=25)

                plt.xlabel("CI")

                for m in range(len(data)):
                  ax.barh(y_pos[m], data[m], color=cmap1(my_norm(cu[m])),
                          edgecolor="#808080", zorder=2)

        if plot_mode == "overlap":
            plt.xlabel("CI and relative CU")

            for m in range(len(data)):
                ax.barh(y_pos[m], data[m], color=color_fill_ci,
                        edgecolor=color_edge_ci, linewidth=1.5, zorder=2)
                ax.barh(y_pos[m], cu[m]*data[m], color=color_fill_cu,
                        edgecolor=color_edge_cu, linewidth=1.5, zorder=2)


        if plot_mode == "combined":
            plt.xlabel("CI and relative CU")

            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('CU', rotation=0, labelpad=25)

            for m in range(len(data)):
                ax.barh(y_pos[m], data[m], color="#ffffff66", edgecolor="#808080", zorder=2)
                ax.barh(y_pos[m], cu[m]*data[m], color=cmap1(my_norm(cu[m])), zorder=2)

        ax.set_facecolor(color="#D9D9D9")

        if not use_influence:
            ax.set_xlim(0, 1)
            ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25/2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(which = 'minor')
        ax.grid(which='minor', color='white')
        ax.grid(which='major', color='white')

        plt.ylabel("Features")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
