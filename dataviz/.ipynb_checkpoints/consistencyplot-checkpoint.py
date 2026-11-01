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
        lst_hsh = sorted(list(cleaned_hsh.items()),key=lambda x: x[0])
        x_data = list(range(0,len(lst_hsh)))
        individual_consistency = []
        sa_gaps = []
        cdata = []
        multi_key = []
        # En cleaned_hsh estan {(id1,id2) : (d,(ti,tj)))}
        for x,y in lst_hsh :
            #print ( ' y is ',  y[1])
            
            k_neighbords = y[1].values
            ic = if_metric(  y[0]  ,k_neighbords)   
            sa_gap = y[2]
            individual_consistency.append( ic )
            multi_key.append ( (x,ic,sa_gap) )

        plot_logic = sorted(multi_key , key=lambda x: x[1])
        consistency  = [y for (x,y,z) in plot_logic]
        #consistency  = [y for (x,y,z) in multi_key]
        #sensgap = [z for (x,y,z) in plot_logic]
        #enumsa = [(x,z) for (x,y,z) in plot_logic]
        #notzero = [(i,a) for (i,a) in enumsa if a!=0]
        #print( 'top 10 non zero ', notzero)
        #fig = make_subplots(specs=[[{"secondary_y": True}]])
        #cdata = ['ID X: '+str(x) + 'SG : ' + str(z) for (x,y,z) in plot_logic   ]
        cdata = ['ID X: '+str(x) + 'SG : ' + str(z) for (x,y,z) in multi_key   ]
        print(' top 10 custgom data ', cdata[0:10])
        #fig.add_trace(go.Scatter(x=x_data,y=consistency,mode='markers',hovertemplate='ID %{x} K-neighbors : %{customdata}   <extra></extra>'),secondary_y=False)
        fig = go.Figure(data=go.Scatter(x=x_data,y=consistency,customdata=cdata,mode='markers',hovertemplate='ID %{x} K-neighbors : %{customdata}   <extra></extra>'))
        #fig.add_trace(go.Scatter(x=x_data,y=sensgap,mode='markers',customdata=cdata,hovertemplate='%{customdata}<extra></extra>'),secondary_y=True,)

        fig.layout.height = 750
        fig.layout.width = 750
        fig.update_layout(xaxis=dict(title=dict(text="Individuals")),yaxis=dict(title=dict(text="Individual consistence")))
        fig.update_layout(xaxis=dict(tickmode='linear',dtick=1))
        fig.update_layout(yaxis=dict(tickmode='linear',tick0=0, dtick=0.25))
        #fig.update_yaxes(range=[0,5], secondary_y=False)
        #fig.update_yaxes(range=[0,5], secondary_y=True)

        return fig
        #fig.update_layout(yaxis=dict(tickmode='linear',tick0=0,dtick=0.25))


'''
        xarr = list(range(0,len(plot_logic)))
        ydata_cons = [y for (x,y,z) in plot_logic ]
        ydata_gap = [z for (x,y,z) in plot_logic ]
        
        total = list(zip(xarr,ydata_cons,ydata_gap))

        consdf = list(zip(ydata_cons,'cons'))

        gapdf = list(zip(ydata_gap,'gap'))

        df1 = pd.DataFrame (data=consdf, columns=['value','type'])
        df2 = pd.DataFrame (data=gapdf, columns=['value','type'])
        df = pd.concat([df1,df2],axis=0)
        fig = px.scatter(df, x="sepal_width", y="value", color="type")'''




        