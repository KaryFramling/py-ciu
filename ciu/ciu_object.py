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

    def plot(self, data, title):
        plt.style.use(self.theme)
        data = data.astype(int)
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.suptitle(title)
        feature_names = self.ci.keys()
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

    def plot_ci(self):
        ci = np.fromiter(self.ci.values(), dtype=float)
        ci = np.round(ci * 100)
        self.plot(ci, 'Contextual Importance')

    def plot_cu(self):
        cu = np.fromiter(self.cu.values(), dtype=float)
        cu = np.round(cu * 100)
        self.plot(cu, 'Contextual Utility')

    def text_explain(self):
        feature_names = self.ci.keys()

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
