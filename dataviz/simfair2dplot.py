

from .attentionscore import fraction_score

from .creator import FairplotCreator
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
class SimilarityFairness2DPlot(FairplotCreator):

    def factory_method(self):
        return SimilarityFairness2DPlot()

    def get_figure(self, cleaned_hsh,attention_score):
        # 1. Generate synthetic 3D data
        x_data = [x[0] for (x, y) in cleaned_hsh]
        y_data = [x[1] for (x, y) in cleaned_hsh]

        spatial_order = sorted(cleaned_hsh, key=lambda row: (row[0][0],  row[0][1]))

        print('spatial order 15 ', spatial_order[0:15])

        scores = { (x[0],x[1]): fraction_score(y[1], y[0]) for (x, y) in spatial_order }
        data = np.zeros((max(x_data)+1, max(y_data)+1))
        auxdata = np.zeros((max(x_data)+1, max(y_data)+1))
        print ( ' data shape ', data.shape)

        for (x, y) in spatial_order:
            data[x[0], x[1]] = fraction_score(y[1], y[0])
            auxdata[x[0], x[1]] = y[1]

        fig = go.Figure(data=go.Heatmap(z=data,customdata=auxdata,colorscale='Reds',hovertemplate='X: %{x}<br>Y: %{y}<br>D: %{customdata} <br>Score: %{z}<extra></extra>'))
        fig.layout.height = 750
        fig.layout.width = 750
        #,hovertemplate='X: %{x}<br>Y: %{y}<br>D: %{customdata[(x,y)][1]} <br>d: %{customdata[(x,y)][0]}   <br>Score: %{z}<extra></extra>'))
        return fig