import plotly.graph_objects as go
from .creator import FairplotCreator
import numpy as np

class SimilarityFairness3DPlot(FairplotCreator):

    def factory_method(self):
        return SimilarityFairness3DPlot()

    # hash = { i: neigbors,targets,gaps  }
    ##### ( target_column[i], targets, gaps[i] )
    #[(0, (0, array([0, 0, 0, 0]), 0)),
    def get_figure(self,cleaned_hsh,if_metric):
        orderedList = sorted ( list(cleaned_hsh.items()) , key=lambda x:x[0])
        print('orderedlist top 5', orderedList[0:5])
        ydata_axis = []
        xdata_axis = []
        zdata_axis = [] 

        for x,y,in orderedList :
            xdata_axis.append( x )
            consist = if_metric(  y[0]  ,y[1] ) 
            ydata_axis.append( consist )
            zdata_axis.append( y[2] )
        print ( ' top y ', ydata_axis[0:5],' top z ', zdata_axis[0:5])
        zipzy = list(zip(ydata_axis , zdata_axis))
        print ( 'top 10 zipzy ' , zipzy[0:10])
        def convert(y,z):
            if y == 1:
                return 1
            elif y < 1 and z > 0:
                return 3
            else:
                return 2
        stroke = [convert(y,z) for (y,z) in zipzy  ]
        print ( ' xdataxis ', xdata_axis[0:10],'ydataxis ', ydata_axis[0:25], 'zdtaxis', zdata_axis[0:25] ,' stroke ',  stroke[0:10])

        fig = go.Figure(data=[go.Scatter3d(
            x=xdata_axis,
            y=ydata_axis,
            z=zdata_axis,
            mode="markers",
            marker=dict(color=stroke, colorscale='reds'),
            hovertemplate=("<b>Point:</b> %{customdata[0]}<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, Sensitive Attribute Gap %{z:.2f})<br><extra></extra>"),
        )])
        fig.update_layout(scene=dict(xaxis=dict(range=(0,13000),title='Row ID'), yaxis=dict(range=(0,1),title='Ind. consistency score'), zaxis=dict(range=(0,5),title='Sens. Attr. Gap')))
        return fig
    '''
    def get_figure(self,cleaned_hsh,attention_score):
        # 1. Generate synthetic 3D data
        np.random.seed(42)
        x_data = [x[0] for (x, y) in list(cleaned_hsh.items())]
        y_data = [x[1] for (x, y) in list(cleaned_hsh.items())]
        z_data = [y[1] / y[0] for (x, y) in list(cleaned_hsh.items())]
        target_diff = [y[1] for (x, y) in list(cleaned_hsh.items())]
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
    '''