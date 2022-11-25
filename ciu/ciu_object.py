from matplotlib import cm, colors, ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CiuObject:
    def __init__(self, ci, cu, c_mins, c_maxs, outval, interactions, theme='fivethirtyeight'):
        self.ci = ci
        self.cu = cu
        self.c_mins = c_mins
        self.c_maxs = c_maxs
        self.outval = outval
        self.interactions = interactions
        self.theme = theme

    def explain_tabular(self):
        out_df = pd.DataFrame.from_dict([self.ci,
                                         self.cu,
                                         self.c_mins,
                                         self.c_maxs])

        out_df.index = pd.Index(['CI', 'CU', 'cmin', 'cmax'], name='Features')
        output_df = out_df.T
        output_df['outval'] = list(self.outval.values())[0]
        return output_df

    @staticmethod
    def _filter_feature_names(feature_names, interactions, include):
        feature_names_final = []
        exclude_interactions = include == 'no_interactions'
        exclude_ordinary = include == 'only_interactions'
        for feature_name in feature_names:
            exclude_feature = \
                (exclude_interactions and feature_name in interactions) or \
                (exclude_ordinary and feature_name not in interactions)
            if not exclude_feature:
                feature_names_final.append(feature_name)
        return feature_names_final
    
    def explain_text(self, include='all', thresholds_ci=None, thresholds_cu=None):
        """
        :param str include: define which features and interactions to use, defaults to 'all'
        :param dict thresholds_ci: dictionary containing the label and ceiling value for the CI thresholds
        :param dict thresholds_cu: dictionary containing the label and ceiling value for the CU thresholds
        :return:
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

        feature_names = self._filter_feature_names(
            self.ci.keys(),
            self.interactions, include
        )

        explanation_texts = []

        for feature in list(feature_names):
            try:
                for k, v in thresholds_ci.items():
                    if self.ci[feature] <= v:
                        ci_text = k
                        break

                for k, v in thresholds_cu.items():
                    if self.cu[feature] <= v:
                        cu_text = k
                        break
            except TypeError:
                raise TypeError(f"The dictionaries containing the CI/CU thresholds cannot have \x1B[3mNone\x1B[0m values. \
                                    \nCI dict: {thresholds_ci} \nCU dict: {thresholds_cu}")


            ci = round(self.ci[feature] * 100, 2)
            cu = round(self.cu[feature] * 100, 2)

            explanation_text = f'The feature "{feature}", which is of ' \
                                   f'{ci_text} (CI={ci}%), is {cu_text} ' \
                                   f'for its prediction (CU={cu}%).'

            explanation_texts.append(explanation_text)

        return explanation_texts

    def plot_ciu(self, plot_mode='default', include='all', sort='ci', color_blind=None,
                 color_fill_ci='#7fffd44d', color_edge_ci='#66CDAA',
                 color_fill_cu="#006400cc", color_edge_cu="#006400"):

        """
        :param str plot_mode: defines the type plot to use between 'default', 'overlap' and 'combined'.
        :param str include: defines whether to include interactions or not.
        :param str sort: defines the order of the plot bars by the 'ci' (default), 'cu' values or unsorted if None.
        :param str color_blind: defines accessible color maps to use for the plots, such as 'protanopia',
                                                        'deuteranopia' and 'tritanopia'.
        :param str color_edge_cu: defines the hex or named color for the CU edge in the overlap plot mode.
        :param str color_fill_cu: defines the hex or named color for the CU fill in the overlap plot mode.
        :param str color_edge_ci: defines the hex or named color for the CI edge in the overlap plot mode.
        :param str color_fill_ci: defines the hex or named color for the CI fill in the overlap plot mode.
        """
        plt.style.use(self.theme)

        data = np.fromiter(self.ci.values(), dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6))
        feature_names_prelim = self.ci.keys()
        feature_names = self._filter_feature_names(
            feature_names_prelim,
            self.interactions, include
        )

        indices_deleted = 0
        for index, feature_name in enumerate(feature_names_prelim):
            if feature_name not in feature_names:
                data = np.delete(data, index - indices_deleted)
                indices_deleted += 1

        y_pos = np.arange(len(feature_names))
        cu = np.fromiter(self.cu.values(), dtype=float)

        if sort == 'ci':
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
                ax.barh(y_pos[m], cu[m] * data[m], color=color_fill_cu,
                        edgecolor=color_edge_cu, linewidth=1.5, zorder=2)

        if plot_mode == "combined":
            plt.xlabel("CI and relative CU")

            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('CU', rotation=0, labelpad=25)

            for m in range(len(data)):
                ax.barh(y_pos[m], data[m], color="#ffffff66", edgecolor="#808080", zorder=2)
                ax.barh(y_pos[m], cu[m] * data[m], color=cmap1(my_norm(cu[m])), zorder=2)

        ax.set_facecolor(color="#D9D9D9")
        ax.set_xlim(0, 1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25 / 2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(which='minor')
        ax.grid(which='minor', color='white')
        ax.grid(which='major', color='white')

        plt.ylabel("Features")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
