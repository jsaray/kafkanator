from .attentionscore import fraction_score
import plotly.graph_objects as go
from .creator import FairplotCreator
import numpy as np

class SimilarityFairness3DPlot(FairplotCreator):

    def factory_method(self):
        return SimilarityFairness3DPlot()

    def get_figure(self,cleaned_hsh,attention_score):
        # 1. Generate synthetic 3D data
        np.random.seed(42)
        x_data = [x[0] for (x, y) in cleaned_hsh]
        y_data = [x[1] for (x, y) in cleaned_hsh]
        z_data = [y[1] / y[0] for (x, y) in cleaned_hsh]
        target_diff = [y[1] for (x, y) in cleaned_hsh]
        zipin = list(zip(z_data, target_diff))
        print('x data ', x_data[0:10], ' y data ', y_data[0:10], 'z data ', z_data[0:10])
        
        # 2. Construct the 3D Scatter Plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode="markers",
            marker=dict(color=[1/fraction_score(y,x) for x, y in zipin], colorscale='reds'),
            hovertemplate=("<b>Point:</b> %{customdata[0]}<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, Simmilarity %{z:.2f})<br><extra></extra>"),
        )])
        
        fig.update_layout(scene=dict(xaxis_title="Dimension X", yaxis_title="Dimension Y", zaxis_title="Dimension Z"),
                          width=900,
                          height=700,
                          margin=dict(l=0, r=0, b=0, t=40))
        print('returning fig')
        return fig