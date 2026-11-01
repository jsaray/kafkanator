
from collections import Counter
import pandas as pd
import random
import copy

def get_counterfactual_values(df,i,sensitive_columns):
    ctf_universe = set( [ tuple(t) for t in df[sensitive_columns].values ] )
    individual = df.iloc[i,:].to_dict()
    indiv_project_sens_cols = ()
    for sens_col in sensitive_columns:
        indiv_project_sens_cols = indiv_project_sens_cols + tuple([int(individual[sens_col])]) 
    universe_list = list(ctf_universe)
    difference_list = [tuple(item) for item in universe_list if item != indiv_project_sens_cols]
    difference_set = set(difference_list)
    return difference_set

def generate_counterfactual_individuals(df,i,sensitive_columns):
    # step 1 : generate the conterfactual universe.
    conterf_indivs = []
    cf_values = get_counterfactual_values(df,i,sensitive_columns)
    ind_df = df.iloc[i,:]
    for c_values in cf_values:
        new_individual = {}
        for (k,v) in ind_df.to_dict().items():
            new_individual[k] = v
        for senscol,cf_value in zip ( sensitive_columns , list(c_values) ):
            new_individual[senscol] = cf_value
        conterf_indivs.append(new_individual)
    df_conterf_indivs = pd.DataFrame(columns=df.columns,data=conterf_indivs)
    return df_conterf_indivs