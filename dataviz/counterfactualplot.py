from kafkanator.dataviz.creator import FairplotCreator
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from kafkanator.dataviz.rankings import rank_by_column
class CounterfactualGapPlot(FairplotCreator) :

    def factory_method(self):
        return ConsistencyPlot()

    def compute_kneighbords(self,id,hsh,k):
        adjas_list = []
        for (x,y) in hsh:
            if x[0] == id:
                adjas_list.append((x,y))
        k_neighbords = sorted ( adjas_list, key=lambda u : u[1][0] )
        return k_neighbords[0:k]

    # Cleaned hsh is a hash containing : 
    # {  (i,j) : (  d , (Ti,Tj)    )  }
    # hDict[i] = (target_column[i], counterf_individuals_matrix , preds_counterf_individuals )
    def get_figure(self, cleaned_hsh,if_metric):
        data = []
        cdata = []
        x_ids = []
        sizes=[]
        std_hsh = sorted(cleaned_hsh.items())
        lines = []
        for(x,y) in std_hsh:
            tobuff = ' x is '+str(x)+' y is '+str(y)
            lines.append(tobuff)
            print ( tobuff )
            x_id = x
            x_target = y[0]
            k_neighbords = y[1]
            ctg = if_metric(  x_target  ,k_neighbords)
            print( ' XTARGET ?', x_target , ' XNEIGB ' , k_neighbords, ' CTG ', ctg )
            data.append( ( x, ctg ))
            cdata = ['XVALUE '+str(x) + 'YVALUE ' + str(y) for (x,y) in data]
            x_ids.append(x_id)
            sizes.append(y[2])
        listx = list(range(len(cleaned_hsh)))

        def rank_by_ctg(x):
            return x[1] 
            
        sortheatmap = [10*y for (x,y) in rank_by_column(data,rank_by_ctg)]
        print ('top 30 sizes ', sizes[0:100])
        print('agrrement ',sortheatmap , 'listx',listx)
        fig = go.Figure(data=go.Scatter(x=listx,y=sortheatmap,customdata=cdata,mode='markers',hovertemplate='ID %{x} K-neighbors : %{customdata}   <extra></extra>'))
        fig.layout.height = 750
        fig.layout.width = 750
        fig.update_layout(xaxis=dict(title=dict(text="Population")),yaxis=dict(tickmode='linear',tick0=0,dtick=1,title=dict(text="Conterfactual Token Fairness Gap")))
        return fig