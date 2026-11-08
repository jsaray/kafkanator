
from collections import Counter
import pandas as pd
import random
import copy

def get_counterfactual_values(df,i,sensitive_columns):
    ''' This method first computes the universe U of possible sensitive column values combinations,
    this is the set { (s1,s2, ... , sn) such that s1 in S1, s2 in S2, ... sn in Sn  } (line 13). 
    After this, build the sensitive attribute Si tuple of individual i (line 16-17).
    After this computes and retrieve set difference between U and the individual sensitive attributes Si, U - Si (line 19).
    The returned value is the effective domain of counterfactual individuals different than i.
    Parameters:
        df an individual of this dataframe will be used for compute counterfactual individuals
        i position of the row we want to compute counterfactual individuals.
        sensitive_columns a set S of sensitive columns .
    Returns
    U - Si
    '''
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
    '''
        This method first compute the set of possible counterfactual sensitive values for an individual (line 35).
        Returned type is a set S of cardinality n.
        After this, the method creates n individuals, each one having sensitive attributes according to one element in 
        element S. The created individuals have non sensitive attributes unchanged.
        
        Parameters :
        df dataframe from individual i originates.
        i position of the individual we are going to compute counterf individuals.
        sensitive_columns the set of sensitive columns.
    '''
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