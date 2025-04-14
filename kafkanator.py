

import matplotlib.pyplot as plt
import numpy as np

def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

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