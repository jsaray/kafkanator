
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