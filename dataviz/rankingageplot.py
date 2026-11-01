from kafkanator.dataviz.attentionscore import score_by_metric,score_by_lambda
import pandas as pd
from kafkanator.dataviz.creator import FairplotCreator
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
import plotly.express as px
class RankingAgePlot(FairplotCreator) :

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
    #hDict[i] = (target_column[i], targets , age  )
    def get_figure(self, cleaned_hsh,if_metric):
        print(' len of hash  ' , len(list(cleaned_hsh.items())))
        datapre = []
        cdata = []
        # En cleaned_hsh estan {(id1,id2) : (d,(ti,tj)))}
        for x,y in cleaned_hsh.items() :
            x_target = y[0]
            k_neighbords = y[1]
            age = y[2]
            datapre.append( ( x, if_metric(  x_target  ,k_neighbords) ,age ) )

        def myrank(t):
            return (t[1],-t[2])

        def get_age_interval(z):
            if 0<=z<=20:
                return 20
            elif 20 < z and z <=60:
                return 60
            elif 60 < z:
                return 100

        order_score_age = score_by_lambda ( datapre ,myrank )
        print ( ' result of ranking by cons score and age ' , order_score_age[0:15])
        by_cons_score_age  = [x for x,y,z in order_score_age  ]
        
        cdata = [(x,z) for x,y,z in order_score_age ]
        ys = [y for x,y,z in order_score_age ]
        print ( ' The DISTINCT consistency score is ', list(set(ys)))
        print('ORDER BY CONSISTENCY AND AGE agrrement ',cdata[0:10])
        #fig = go.Figure(data=go.Scatter(x=list(range(0,len(order_score_age))),y=ys,customdata=cdata,hovertemplate='ID %{customdata}   <extra></extra>'))
        visuhsh = [(x,y,get_age_interval(z)) for x, y ,z in order_score_age]
        l_icicle = []
        for (x,y,z) in order_score_age:
            row = {'individual':x,'consistency_score': y , 'age_interval' : get_age_interval(z) , 'population' : 1  }
            l_icicle.append(row)
        
        df_icicle = pd.DataFrame(data=l_icicle,columns=['individual','consistency_score','age_interval','population'])
        #df_icicle.to_csv ( 'kfkntr_output_barstack.csv' ) 
        result = df_icicle.groupby(['consistency_score', 'age_interval'])['population'].sum().reset_index()
        ordered_result = result.sort_values(by=['consistency_score','age_interval'])
        #print('the types ', result.dtypes)
        print(' top 10 ordered result ', ordered_result.iloc[0:10,:])
        fig = px.bar(ordered_result, x="consistency_score", y="population", color="age_interval",title="Individual consistency score and age distribution", color_continuous_scale=["orange", "red", "green", "blue","purple"])
        #fig = px.icicle(df_icicle, path=[px.Constant("Consistency score"), 'consistency_score', 'age_interval'], values='population',color='age_interval',color_continuous_scale='RdBu')
        
        
        fig.layout.height = 750
        fig.layout.width = 750

        fig.write_image("fig1.png")
        #fig.update_layout(xaxis=dict(title=dict(text="Individuals")),yaxis=dict(title=dict(text="Individual consistence")))
        #fig.update_layout(xaxis=dict(tickmode='linear',tick0=0,dtick=1000))
        return fig