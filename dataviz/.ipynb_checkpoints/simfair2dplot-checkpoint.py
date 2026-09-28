

from fairplot import FairPlot
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

class SimilarityFairness2DPlot(FairPlot):

    def __init__(self, cleaned_hsh,attention_score):
        self.cleaned_hsh = cleaned_hsh
        self.attention_score = attention_score

    def get_figure(self):
        # 1. Generate synthetic 3D data
        x_data = [x[0] for (x, y) in self.cleaned_hsh]
        y_data = [x[1] for (x, y) in self.cleaned_hsh]
        data = np.zeros(len(x_data),len(y_data))

        for (x,y) in self.cleaned_hsh:
            data[x][y] = self.attention_score(x,y)
            
        fig = px.imshow(data, color_continuous_scale='reds', origin='lower')
        return fig