from .attentionscore import fraction_score
import plotly.graph_objects as go
from .creator import FairplotCreator
import numpy as np

class SimilarityRankingTable(FairplotCreator):

    def factory_method(self):
        return SimilarityRankingTable()
    
    def get_figure(self,cleaned_hsh,attention_score):
        table_data = sorted([ ( x[0], x[1] , int(attention_score(y[1], y[0])) ) for (x, y) in cleaned_hsh ], key=lambda row: row[2], reverse=True)
        l_table_data = [list(row) for row in table_data]
        t_table_data = list(map(list, zip(*l_table_data)))
        print('table data ', l_table_data[0:10])
        fig = go.Figure(data=[go.Table(header=dict(values=['Row X', 'Row Y', 'Attention Score']),
                                      cells=dict(values=t_table_data[0:10]))])
        return fig