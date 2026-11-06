import pandas as pd
import plotly.express as px
def if_stacked_bar( df, x_column , aggregate_column , group_by_attribute, title, color_continous_scale=[] ):
        
    df_icicle = pd.DataFrame(data=df,columns=['individual','consistency_score','age_interval','population'])
        #df_icicle.to_csv ( 'kfkntr_output_barstack.csv' ) 
    result = df_icicle.groupby(['consistency_score', 'age_interval'])['population'].sum().reset_index()
    ordered_result = result.sort_values(by=['consistency_score','age_interval'])
        #print('the types ', result.dtypes)
        #print(' top 10 ordered result ', ordered_result.iloc[0:10,:])
    fig = px.bar(ordered_result, x=x_column, y=aggregate_column, color=group_by_attribute,
                 title=title, color_continuous_scale=color_continous_scale)

    return fig