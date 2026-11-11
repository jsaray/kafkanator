import plotly.graph_objects as go
from .creator import FairplotCreator
import numpy as np
from .rankings import rank_by_column
class SimilarityFairness3DPlot(FairplotCreator):

    def factory_method(self):
        return SimilarityFairness3DPlot()

    # hash = { i: neigbors,targets,gaps  }
    ##### ( target_column[i], targets, gaps[i] )
    #[(0, (0, array([0, 0, 0, 0]), 0)),
    def get_figure(self,cleaned_hsh,if_metric):
        #orderedList = sorted ( list(cleaned_hsh.items()) , key=lambda x:x[0])
        def rank_by_x (t):
            return t[0]
        orderedList = rank_by_column( list(cleaned_hsh.items()) , rank_by_x )
        print('orderedlist top 5', orderedList[0:5])
        ydata_axis = []
        xdata_axis = []
        zdata_axis = [] 
        cdata = []
        for x,y,in orderedList :
            xdata_axis.append( x )
            consist = if_metric(  y[0]  ,y[1] ) 
            ydata_axis.append( consist )
            zdata_axis.append( y[2] )

        cdata=[ 'XVALUE '+str(x) for (x,y) in orderedList ]
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
            customdata=cdata,
            mode="markers",
            marker=dict(color=stroke, colorscale='reds'),
            hovertemplate=cdata,
        )])
        fig.update_layout(scene=dict(xaxis=dict(range=(0,13000),title='Row ID'), yaxis=dict(range=(0,1),title='Ind. consistency score'), zaxis=dict(range=(0,5),title='Sens. Attr. Gap')))
        return fig