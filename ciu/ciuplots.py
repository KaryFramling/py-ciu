import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

