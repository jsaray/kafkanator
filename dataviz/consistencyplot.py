from .attentionscore import score_by_metric

from .creator import FairplotCreator
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
from plotly.subplots import make_subplots
import pandas as pd
class ConsistencyPlot(FairplotCreator) :

    def factory_method(self):
        return ConsistencyPlot()

    # Cleaned hsh is a hash containing : 
    # {  (i,j) : (  d , (Ti,Tj)    )  }
    #hDict[i] = (y_enc[i], targets, g   )
    def get_figure(self, cleaned_hsh,if_metric):
        k = 4
        individual_consistency = []
        sa_gaps = []
        cdata = []
        multi_key = []
        # En cleaned_hsh estan {(id1,id2) : (d,(ti,tj)))}
        for x,y in cleaned_hsh.items() :
            k_neighbords = y[1].values

            ic = if_metric(  y[0]  ,k_neighbords)   
            sa_gap = y[2]
            individual_consistency.append( ic )
            multi_key.append ( (x,ic,sa_gap) )

        plot_logic = sorted(multi_key , key=lambda x: x[1])
        #print (' JUST AFTER PLOT ', plot_logic)
        x_data = list(range(len(plot_logic)))
        consistency  = [y for (x,y,z) in plot_logic]
        cdata = ['XVALUE '+str(x) for (x,y,z) in plot_logic]
        #print(' top 10 custom data ', cdata[0:10])
        fig = go.Figure(data=go.Scatter(x=x_data,y=consistency,customdata=cdata,mode='markers',hovertemplate='%{customdata}   <extra></extra>'))
        fig.layout.height = 750
        fig.layout.width = 750
        fig.update_layout(xaxis=dict(title=dict(text="Individuals")),yaxis=dict(title=dict(text="Individual consistence")))
        fig.update_layout(xaxis=dict(tickmode='linear',dtick=1))
        fig.update_layout(yaxis=dict(tickmode='linear',tick0=0, dtick=0.25))
        return fig
