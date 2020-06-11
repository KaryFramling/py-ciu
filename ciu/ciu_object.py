import matplotlib.pyplot as plt
import numpy as np


class CiuObject:
    def __init__(self, ci, cu, interactions, theme='fivethirtyeight'):
        self.ci = ci
        self.cu = cu
        self.interactions = interactions
        self.theme = theme

    @staticmethod
    def _determine_importance(ci):
        if ci < 0.25:
            return 'not important'
        if ci < 0.5:
            return 'important'
        if ci < 0.75:
            return 'very important'
        else:
            return 'highly important'

    @staticmethod
    def _determine_typicality(cu):
        if cu < 0.25:
            return 'not typical'
        if cu < 0.5:
            return 'unlikely'
        if cu < 0.75:
            return 'typical'
        else:
            return 'very typical'

    @staticmethod
    def _filter_feature_names(feature_names, interactions, include):
        feature_names_final = []
        for index, feature_name in enumerate(feature_names):
            exclude_interactions = include == 'no_interactions'
            exclude_ordinary = include == 'only_interactions'
            exclude_feature = \
                (exclude_interactions and feature_name in interactions) or \
                (exclude_ordinary and feature_name not in interactions)
            if not exclude_feature:
                feature_names_final.append(feature_name)
        return feature_names_final

    def plot(self, data, title, include):
        plt.style.use(self.theme)
        data = data.astype(int)
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.suptitle(title)
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
        ax.barh(y_pos, data)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xticks([])
        for index, y in enumerate(data):
            ax.text(15, index - 0.1, y, fontsize='large', size=10, ha='right',
                    va='bottom')
            ax.text(21, index - 0.1, '%', fontsize='large', size=10, ha='right',
                    va='bottom')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)

    def plot_ci(self, include='all'):
        ci = np.fromiter(self.ci.values(), dtype=float)
        ci = np.round(ci * 100)
        self.plot(ci, 'Contextual Importance', include)

    def plot_cu(self, include='all'):
        cu = np.fromiter(self.cu.values(), dtype=float)
        cu = np.round(cu * 100)
        self.plot(cu, 'Contextual Utility', include)

    def text_explain(self, include='all'):
        feature_names = self._filter_feature_names(
            self.ci.keys(),
            self.interactions, include
        )
        explanation_texts = []
        for index, feature in enumerate(list(feature_names)):
            importance = self._determine_importance(self.ci[feature])
            typicality = self._determine_typicality(self.cu[feature])
            ci = round(self.ci[feature] * 100, 2)
            cu = round(self.cu[feature] * 100, 2)
            explanation_text = f'The feature "{feature}", which is ' \
                               f'{importance} (CI={ci}%), is {typicality} ' \
                               f'for its class (CU={cu}%).'
            explanation_texts.append(explanation_text)

        return explanation_texts
