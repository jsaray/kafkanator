from .fairness.simmilarity import simmilarity_fairness_hash
from sklearn.calibration import calibration_curve, CalibrationDisplay
import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from dash import dash_table
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import brier_score_loss
from .calibration.metrics import ece

def simmilarity_fairness_3d( cleaned_hsh ) :
    # 1. Generate synthetic 3D data
    np.random.seed(42)
    x_data = [ x[0] for (x,y) in cleaned_hsh ]
    y_data = [ x[1] for (x,y) in cleaned_hsh ]
    z_data = [ y[0] for (x,y) in cleaned_hsh]
    print ( 'x data ', x_data[0:10],' y data ',y_data[0:10] ,'z data ' , z_data[0:10] )
    # 2. Construct the 3D Scatter Plot
    fig = go.Figure(data=[go.Scatter3d(x=x_data,
                                   y=y_data,
                                   z=z_data,
                                   mode="markers",
                                   marker=dict(color=[1/x for x in z_data],colorscale='reds'),
                                   hovertemplate=("<b>Point:</b> %{customdata[0]}<br><b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, Simmilarity %{z:.2f})<br><extra></extra>"),
            )
        ]
    )
    fig.update_layout(scene=dict(xaxis_title="Dimension X", yaxis_title="Dimension Y", zaxis_title="Dimension Z"),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=40))
    print('returning fig')
    return fig

def similar_subjects_treatment_plot( data, sensitive_column, sensitive_attribute_values ,numrows,target_column,simmilarity_distance='gower'):
    '''
    This method produce a 3d plot , (X,Y) corresponds to individuals in different set partitions . 
    Z coordinates show such pairs that being simmilar, had different treatment by your model. The closer to the
    plane Z=0, the more simmilar they are so the more attention you must pay for different treatment in your model.
    
    Parameters : 
    data : a dataframe containing a sensitive column that contains a PAIR of sensitive attribute values, 
    sensitive_column : the sensitive column.
    sensitive_attribute_values : an array with TWO possible values . 
    simmilarity_attr_hsh : for the moment we are using a simmilarity distance based on adding up 1 whenver two categorical columns of two rows are different, and adding up 
    absolute value if they are two numerical columns.
    simmilarity_distance only the value catnum_simmilarity_distance from the moment

    NOTE THAT UP TO NOW IS FROM 1 TO 100 ROWS, MUST BE CHANGED !!!!!!
    '''
    hsh = simmilarity_fairness_hash( data , sensitive_column , sensitive_attribute_values ,numrows, target_column )
    cleaned_hsh = [(x,y) for (x,y) in hsh if y[1] == False and y[0]<= 0.1]
    figu = simmilarity_fairness_3d( cleaned_hsh )
    return figu

def similar_subjects_dashboard( data, sensitive_column,sensitive_attribute_values,numrows,target_column,simmilarity_distance='gower'):
    simplot = similar_subjects_treatment_plot( data , sensitive_column ,sensitive_attribute_values ,  numrows ,target_column)
    app = dash.Dash(__name__)
    # Define the layout with two columns
    app.layout = html.Div(
        style={'display': 'flex', 'height': '100vh','padding': '20px','flexDirection': 'column' },
        children=[
            # Column 1: Graph
            html.Div(
                children=[
                    html.H1("Simmilar Subjects Plot")]
            ),
            html.Div(
                style={'overflow-x': 'auto','vertical-align':'center'},
                children=[
                    html.H3("Data for Hovered Point(s)"),
                    html.Div(id='table-container', children=[
                        html.P("Hover over a point in the 3D scatter plot to see related data.")
                ])
            ]
            ),
            html.Div(
                children=[
                    dcc.Graph(
                        id='scatter3d',
                        figure=simplot,
                        style={'height': '50vh'}
                    )
                ]
            )
        ]
    )
    # Callback to update the table based on hover data
    @app.callback(
        Output('table-container', 'children'),
        Input('scatter3d', 'hoverData')
    )
    def update_table(hoverData):
        if hoverData is None:
            return html.P("Hover over a point in the 3D scatter plot to see related data.")
        reo = np.append(['ID'],data.columns) 
        reorder = np.append(reo,['Target'])
        completeD = data.iloc[0:numrows,:].assign(Target=target_column)
        completeData = completeD.assign(ID=completeD.index)
        completeData = completeData.reindex(reorder,axis='columns')
        x=hoverData['points'][0]['x']
        y=hoverData['points'][0]['y']
        z=hoverData['points'][0]['z']
        subp1 = completeData[completeData[sensitive_column] == sensitive_attribute_values[0] ]
        subp2 = completeData[completeData[sensitive_column] == sensitive_attribute_values[1] ]
        row1 = subp1[subp1['ID'] == x].iloc[0,:]
        row2 = subp2[subp2['ID'] == y].iloc[0,:]
        datatod = [row1.to_dict(),row2.to_dict()]
        allc = completeData.columns
        data_t = dash_table.DataTable( style_table={'overflowX': 'auto'},columns = [{"name": i, "id": i} for i in completeData.columns] , data=datatod)
        try:
            return html.Div([data_t])
        except (IndexError, KeyError):
            return html.P("Error retrieving data for the hovered point.")
    return app

def reliability_diagram_plot ( data,sensitive_column,sensitive_attribute_values,predicted_column,real_column,bins=10 ):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(1, 1)
    colors = plt.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:1, :1])
    calibration_displays = {}
    ecestr = []
    brierstr = []
    for sensitive_attribute in sensitive_attribute_values:
        subp = data[  data[sensitive_column] == sensitive_attribute ]
        subp_real = subp[ real_column ]
        subp_preds = subp[predicted_column]
        prob_true_subp, prob_pred_subp = calibration_curve(subp_real, subp_preds, n_bins=10)
        ece_val = ece(list(zip(subp_preds,subp_real)),n_bins=10)
        #print ( 'true ', prob_true_subp[0:10] , ' pred ', prob_pred_subp[0:10])
        brier_val = brier_score_loss(subp_real, subp_preds ,pos_label=1)
        eces = ' ECE ' + str(sensitive_attribute) + "=" + str(ece_val) + ' ' 
        ecestr.append(eces)

        briers = ' BRIER ' + str(sensitive_attribute) + "=" + str(brier_val) + ' ' 
        brierstr.append(briers)
        display = CalibrationDisplay.from_predictions(subp_real, subp_preds,n_bins=10,ax=ax_calibration_curve,name=sensitive_attribute)
        calibration_displays[sensitive_attribute] = display

    ax_calibration_curve.grid()
    title = "Reliability Diagrams " + ' '.join(ecestr) + '\n' + ' '.join(brierstr)
    ax_calibration_curve.set_title(title)    
    return fig

