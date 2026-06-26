from .fairness.simmilarity import simmilarity_fairness_hash
import plotly.graph_objects as go
import numpy as np
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
    fig.update_layout(title="Interactive 3D Scatter Plot with Custom Tooltips",scene=dict(xaxis_title="Dimension X", yaxis_title="Dimension Y", zaxis_title="Dimension Z"),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=40))
    return fig

def similar_subjects_treatment_plot( data, sensitive_column, sensitive_attribute_values ,numrows,attr_sim_compas,simmilarity_distance='catnum_simmilarity_distance'):
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
    hsh = simmilarity_fairness_hash( data , sensitive_column , sensitive_attribute_values ,attr_sim_compas,numrows )
    cleaned_hsh = [(x,y) for (x,y) in hsh if y[1] == False and y[0] <= 5 ]
    figu = simmilarity_fairness_3d( cleaned_hsh )
    figu.show()