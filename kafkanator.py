

import matplotlib.pyplot as plt
import numpy as np

'''
Computes the gini index from a ascending order gains array
param x an array containing gains sorted in ascending order
for example [1,1,2,2,3,3,3] means a population of 7 people, the first one
gain is 1, the last one 3 and so on.
return gini index of this array
'''
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

'''
Computes robin hood index. This is the percentage of income
tha must be redistributed in population, in order to be egalitarian.

parameters : x a vector representing the gains of the population.
i.e [5,3,5,6,9] means that one person has 5 gains, the next one three and so on.
Total gain will be sum(x)

return : a number between 0 and 1 meaning the percentage of the income that must
be redistributed. A number close to 1 means high concentration of wealth in few hands
a number close to 0 means a distribution of wealth close to egalitarian state. 
'''
def robin_hood(income_array):
    egal_income = sum(income_array) / len(income_array)
    deltas = []
    for inc in income_array:
        if egal_income - inc < 0 :
            deltas.append(abs(egal_income - inc))
    rh_index = sum(deltas) / sum(income_array)
    return rh_index


def theil_index_L(income_array):
    x_mean = sum(income_array)/len(income_array)
    divided_mean = [ np.log(x_mean / i) for i in income_array ]
    summin = sum(divided_mean)
    return summin/len(income_array)


def theil_index_T(income_array,array_type='props',base_entropy=np.e):
    if array_type == 'props':
        return np.log(len(income_array)) - entropy(income_array,base=base_entropy)
    elif array_type == 'gains':
        props_array = np.array([x/sum(income_array) for x in income_array])
        return np.log(len(income_array)) - entropy(props_array,base=base_entropy)


'''
This function computes the lorentz curve coordinates from a population and income array
param: population an array containing in position i a number representing the amount of
people earning income[i]
param: income an array containing in position i a number representing the earning of the people
in population[i].
param: gini flag to True if you want the gini computed on the position 3 of returning tuple.
Example lorentz_curve ( [50,20,30,10],[100,300,200,30]) means that 50 people earn 100, 20 people earn 300 and so on.
return 2-tuple with lorentz_curve coordinates to be plotted using the visual framework of your choice. If you set gini flag to
true, it wil be a 3-tuple, in the third position you find the gini coefficient.
'''
def lorentz_curve ( population , income ,gini_index=False):
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

def gini_per_cluster(df,gb_column,income_column):
    salary_groups = df.groupby ([gb_column]) 
    setOfCats = set(df[gb_column].values)
    ginis = []
    for s in setOfCats:
        subgroup = df.iloc[salary_groups.groups[s],:]
        incomes = sorted(subgroup[income_column].values)
        print ( 'sorted incomes ',s, ' ', incomes)
        g = gini(incomes)
        ginis.append((s,g))
    return ginis