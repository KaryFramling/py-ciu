import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ciu.CIU import contrastive_ciu

def ciu_beeswarm(df, xcol='CI', ycol='feature', color_col='norm_invals', legend_title=None, jitter_level=0.5, 
                 palette = ["blue", "red"], opacity=0.8):
    """
    Create a beeswarm plot of values. This can be used for CI, Cinfl, CU or any values in principle 
    (including Shapley value, LIME values, ...).

    **Remark:** This has not been tested/implemented for non-numerical values, intermediate concepts etc.
    (unlike the R version). 

    :param: df: A "long" CIU result DataFrame, typically produced by a call to :func:`ciu.CIU.CIU.explain_all`. 
    :type df: DataFrame
    :param xcol: Name of column to use for X-axis (numerical).
    :type xcol: str
    :param ycol: Name of column to use for Y-axis, typically the one that contains feature names.
    :type ycol: str
    :param color_col: Name of column to use for dot color, typically the one that instance/feature values 
        that are normalised into `[0,1]` interval.
    :type color_col: str
    :param legend_title: Text to use as legend title. If `None`, then used `color_col`.
    :type legend_title: str
    :param jitter_level: Level of jitter to use.
    :type jitter_level: float
    :param palette: Color palette to use. The default value is a list with two colors but can probably be 
        any kind of palette that is accepted by plotly.graphobjects.
    :type palette: list
    :param opacity: Opacity value to use for dots.
    :type opacity: float


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
    Create a contrastive plot for the two CIU results passed. This is essentially similar to 
    an influence plot. 

    :param ciures1: See :func:`ciu.CIU.contrastive_ciu`
    :type ciures1: DataFrame
    :param ciures2: See :func:`ciu.CIU.contrastive_ciu`
    :type ciures2: DataFrame
    :param xminmax: Min/max values to use for X axis.
    :type xminmax: array/list
    :param main: Main title to use.
    :type main: str
    :param figsize: Figure size.
    :type figsize: array
    :param colors: Bar colors to use.
    :type colors: array
    :param edgecolors: Bar edge colors to use.
    :type edgecolors: array

    :return: A pyplot plot.
    """
    contrastive = contrastive_ciu(ciures1, ciures2)
    feature_names = ciures1['feature']
    nfeatures = len(feature_names)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(nfeatures)

    #    cinfl, feature_names = (list(t) for t in zip(*sorted(zip(cinfl, feature_names))))

    plt.xlabel("ϕ")
    for m in range(len(contrastive)):
            ax.barh(y_pos[m], contrastive.iloc[m], color=[colors[0] if contrastive.iloc[m] < 0 else colors[1]],
                    edgecolor=[edgecolors[0] if contrastive.iloc[m] < 0 else edgecolors[1]], zorder=2)

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


