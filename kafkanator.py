

import matplotlib.pyplot as plt
import numpy as np

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



def lorentz_curve ( population , income ):
    perc_population = np.array([x/sum(population) for x in population])
    perc_income = np.array([x/sum(income) for x in income])
    cum_perc_pop = np.concatenate(([0], perc_population.cumsum()), axis=None)
    cum_perc_inc = np.concatenate(([0],perc_income.cumsum()), axis=None)
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