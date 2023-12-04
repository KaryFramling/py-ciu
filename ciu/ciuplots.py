import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ciu.CIU import contrastive_ciu

def ciu_beeswarm(df, xcol='CI', ycol='feature', color_col='norm_invals', legend_title=None, jitter_level=0.5, 
                 palette = ["blue", "red"], opacity=0.8):
    """
    Create a beeswarm plot.

    :param: df: A CIU result DataFrame, typically produced by a call to `explain_all()`. 

    :return: A plotly.graphobjects Figure.
    """

    # Deal with None parameters
    if legend_title is None:
        legend_title = color_col

    N = len(df)
    fig = go.Figure()
  
    features = pd.Categorical(df.loc[:,ycol]).categories
    nfeatures= len(features)
    for i in range(nfeatures):
        f = features[i]
        inds = np.where(df.loc[:,ycol]==f)[0]
        dfs = df.iloc[inds,:]
        marker = dict(
            size=9,
            color=dfs[color_col],
            colorscale=palette,
            opacity=opacity,
        )
        if i == 0:
            marker['colorbar'] = dict(title=legend_title)
        fig.add_trace(go.Scatter(
            x=dfs.loc[:,xcol], 
            y=i + np.random.rand(N) * jitter_level,
            mode='markers',
            marker=marker,
            name=f,
        ))
    fig.update_layout(showlegend=False, coloraxis_showscale=True, legend_title_text='My Legend Title')
    fig.update_yaxes(tickvals=list(range(len(features))), ticktext=list(features))
    return fig

def plot_contrastive(ciures1, ciures2, xminmax=None, main=None, figsize=(6, 4), 
                     colors=("firebrick","steelblue"), edgecolors=("#808080","#808080")):
    """
    Create a contrastive plot for the two CIU results passed.
    """
    contrastive = contrastive_ciu(ciures1, ciures2)
    feature_names = ciures1['feature']
    nfeatures = len(feature_names)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(nfeatures)

    #    cinfl, feature_names = (list(t) for t in zip(*sorted(zip(cinfl, feature_names))))

    plt.xlabel("ϕ")
    for m in range(len(contrastive)):
            ax.barh(y_pos[m], contrastive[m], color=[colors[0] if contrastive[m] < 0 else colors[1]],
                    edgecolor=[edgecolors[0] if contrastive[m] < 0 else edgecolors[1]], zorder=2)

    plt.ylabel("Features")
    if xminmax is not None:
        ax.set_xlim(xminmax)
    if main is not None:
        plt.title(main)

    ax.set_facecolor(color="#D9D9D9")

    # Y axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.grid(which = 'minor')
    ax.grid(which='minor', color='white')
    ax.grid(which='major', color='white')


