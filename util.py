import numpy as np
def categorize_interval(df, column, diction ):
    """This method applies a function on a pandas data frame column, the goal is to replace integer intervals by a categorical values.

    Args:
        df (pandas DataFrame): dataframe .
        column (str): The column on we will apply the number to category transformation.
        diction (str):  A dictionary containing the number - category mapping. For example if we want to convert age intervals to human readable : { 'young':(20,30) , 'adult':(31,50), 'elder':(51,70) }
    
    Returns:
        DataFrame : a dataframe with numerical column 'column' transformed into a categorical one according to diction mapping.
    """
    # dictio = { 'young':[20,30] , 'adult':[31,50], 'elder':[51,70]  }
    def interval_function(x,dictio):
        for (k,v) in dictio.items():
            if v[0] <= int(x) and int(x) <= v[1] :
                return k
    return df[column].apply(lambda x: interval_function(x,diction))

def rename_column ( df, column , dictionn ):
    def rename_function(x,diction):
        return diction[x]
    return df[column].apply(lambda x : rename_function(x,dictionn))


def transform_dict_keys_to_str(sp):
    ret_dict = {}
    for (k,v) in sp.items():
        newk = ','.join([str(y) for y in k])
        ret_dict[newk] = v
    return ret_dict

def compareSubstraction(d,s):
    if 0 <= d and d <= 0.1 :
        return ['background-color : green' for v in s]
    elif 0.1 <= d and d <= 0.2 :
        return ['background-color : yellow' for v in s]
    elif 0.2 <= d  :
        return ['background-color : red' for v in s ]

def compareRatio(d,s):
    if 0 <= d and d <= 0.5 :
        return ['background-color : red' for v in s]
    elif 0.5 <= d and d <= 0.8 :
        return ['background-color : yellow' for v in s]
    elif 0.8 <= d  :
        return ['background-color : green' for v in s ]
    
def default_row_highlighting(s):
    """In house method to beautify fairness metrics dataframe providing a 3 color palette (green,yellow,red) depending on 
    the fairness gap in your model.

    Args:
        s DataFrame row.
    """
    inde = s.index
    d = abs( float(s.loc['0']) - float(s.loc['1'] ))
    if 'PREVALENCE' in inde :
        return compareRatio(d,s)
    else:
        print ( 'd is ', str(d))
        if type(d) == float or isinstance(d,np.floating ):
            print ( ' in if')
            return compareSubstraction(d,s)