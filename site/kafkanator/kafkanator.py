import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
import pandas as pd

def index_on_dataframe_column(df: pd.DataFrame, column: str, index_function: callable, **kwargs ) -> float :
    """This method computes a unequality index over a pandas dataframe column.

    Args:
    df (pandas DataFrame): The dataframe
    column (str): The column we will apply the inequality
    index_function (callable): the index we want to apply, is a kafkanator function such as Gini, robin_hood, theil L or theil T.
    kwargs (dict), auxiliar parameters the index_function could use, for example if theil_t, we can input here the base.
    
    Returns:
        float: representing the inequality index.
    """
    sorted_df = df.sort_values(by=[column])
    return index_function( np.array(sorted_df[column].values ) , **kwargs )

def gini(x):
    """Computes the gini index from an ascending order gains array.

    Args:
    x (list): Gains sorted in ascending order, for example [1,1,2,2,3,3,3] means a population of 7 people, the first one
    gain is 1, the last one 3 and so on.
    
    Returns: 
        float : The Gini index for this array
    """
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

def robin_hood(income_array):
    """Computes robin hood index. This is the percentage of income
    tha must be redistributed in population in order to be egalitarian.

    Args:

    x (list): Represents population gains, i.e [5,3,5,6,9] means that one person has 5 gains, the next one three and so on.
    Total gain will be sum(x), total population will be len(x)

    Returns:
        float : A number between 0 and 1, is the percentage of the income that must be redistributed. A number close to 1 means high concentration of wealth in few hands
    a number close to 0 means a distribution of wealth close to egalitarian state. 
    """
    egal_income = sum(income_array) / len(income_array)
    deltas = []
    for inc in income_array:
        if egal_income - inc < 0 :
            deltas.append(abs(egal_income - inc))
    rh_index = sum(deltas) / sum(income_array)
    return rh_index

def theil_index_L(income_array):
    """Computes the Theil L index.

    Args:

    income_array (list): array of incomes.
    
    Returns: 
        float: the theil L index.
    """
    x_mean = sum(income_array)/len(income_array)
    divided_mean = [ np.log(x_mean / i) for i in income_array ]
    summin = sum(divided_mean)
    return summin/len(income_array)

def theil_index_T(income_array,array_type='props',base_entropy=np.e):
    """Computes the Theil T index.

    Args:
    
    income_array (list): array of incomes.
    array_type (str): if 'props'  this means all numbers in income_array are between 0 and 1, and all must sum up to 1. if 'gains' this means income_array are integers representing gains.
    base_entropy (float) the base to compute the entropy, remember that entropy is a family of functions with diferent bases, e constant by default,
    
    Returns: 
        float : the theil T index
    """
    if array_type == 'props':
        return np.log(len(income_array)) - entropy(income_array,base=base_entropy)
    elif array_type == 'gains':
        props_array = np.array([x/sum(income_array) for x in income_array])
        return np.log(len(income_array)) - entropy(props_array,base=base_entropy)

def lorentz_curve ( population , income ,gini_index=False):
    """This function computes the lorentz curve coordinates from a population and income array.

    Args:

    population (list): contains in position i a number representing the amount of people earning income[i]
    income (list): contains in position i a number representing the earning of the people in population[i].
    gini (boolean) True if you want the gini computed on the position 3 of returning tuple. False otherwise.
    Example lorentz_curve ( [50,20,30,10],[100,300,200,30]) means that 50 people earn 100, 20 people earn 300 and so on.
    
    Returns:
        tuple: 2-tuple with lorentz_curve coordinates to be plotted using the visual framework of your choice. If you set gini flag to
    true, it wil be a 3-tuple, in the third position you find the gini coefficient.
    """
    assert( len(population) == len(income))
    zippedSortedArray = sorted( list( zip( population , income) ) , key= lambda x: x[1] )
    print ( ' sorted array ', zippedSortedArray)
    perc_population = np.array([p/sum(population) for (p,g) in zippedSortedArray])
    perc_income = np.array([g/sum(income) for (p,g) in zippedSortedArray])
    cum_perc_pop = np.concatenate(([0], perc_population.cumsum()), axis=None)
    cum_perc_inc = np.concatenate(([0],perc_income.cumsum()), axis=None)
    if gini_index:
        arrayGini = []
        for (p,g) in zippedSortedArray:
            arrayGini = np.concatenate( ( arrayGini , np.repeat(g,p)  ) )
        print ( ' gini input ', arrayGini )
        g_index = gini(np.array(arrayGini))
        return (cum_perc_pop,cum_perc_inc,g_index)
    else:
        return (cum_perc_pop,cum_perc_inc)

def index_per_cluster(df,group_by_column,income_column,index='gini', **kwargs ):
    """Make clusters over a data frame and apply an index on each of them .

    Args:
    df (pandas Dataframe): a data frame where you have data about gains to be divided according to a column.
    group_by_column (str): the column you will perform your group by on.
    income_column (str): column where you have the gains/incomes. For the moment the column must have numeric integer values, not proportions.
    index (str): the type of inequality index you will use , you have gini, theil-t , theil-l, and robin hood.
    kwargs (dict): optional, used in case you use theil - T, you can put here auxiliar parameter such as entropy base. 

    Returns:
        list: an array of tuples, each tuple is a value of the group_by_column, followed by the intra cluster index of your choice.
    """
    salary_groups = df.groupby ([group_by_column]) 
    setOfCats = set(df[group_by_column].values)
    indexes = []
    for s in setOfCats:
        subgroup = df.iloc[salary_groups.groups[s],:]
        incomes = sorted(subgroup[income_column].values)
        print ( 'sorted incomes ',s, ' ', incomes)
        if index == 'gini':
            g = gini(incomes)
        elif index == 'theil-t':
            g = theil_index_T ( incomes,**kwargs )
        elif index == 'theil-l':
            g = theil_index_L ( incomes )               
        elif index == 'robin-hood':
            g = robin_hood ( incomes )
        indexes.append((s,g))
    return indexes