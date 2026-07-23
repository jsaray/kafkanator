import gower
import pandas as pd

def same_prediction( ind1 , ind2 ,target_column):
    samePrediction = True if ind1[target_column] == ind2[target_column] else False
    return samePrediction

def simmilarity_fairness_hash( data, sensitive_column, sensitive_attribute_values ,numrows,target_column,simmilarity_distance='gower'  ):
    # For the moment, only 2 values in sensitive_attribute_values allowed. if more than 2 we have to do like the correlation plots
    hDict = {}
    sub_matrix = data.iloc[0:numrows,:]
    wo = sub_matrix[ sub_matrix[sensitive_column] == sensitive_attribute_values[0]]
    wo_target = target_column[ wo.index ]
    print ( ' wo target ', wo_target, ' dtype ', wo_target.dtype)
    ma = sub_matrix[ sub_matrix[sensitive_column] == sensitive_attribute_values[1]]
    ma_target = target_column[ ma.index ]
    print ( ' ma target ', ma_target, ' dtype ', ma_target.dtype)
    print('getting distance matrix, data shape ')
    if simmilarity_distance == 'gower':
        distance_matrix = gower.gower_matrix(sub_matrix.iloc[0:numrows,:])
    print('finish distance matrix')
    
    list_subp1 = list(wo.index)
    list_subp2 = list(ma.index)
    #print ( 'list 1 ', list_subp1 , ' list 2 ', list_subp2)
    for (i,a) in enumerate(distance_matrix):
        for j,e in enumerate(a):
            if i in list_subp1 and j in list_subp2:
                reto = True if wo_target[i] == ma_target[j] else False
                if reto == False:
                    hDict[(i,j)] = (distance_matrix[i][j],reto)

    sor = sorted(hDict.items(), key=lambda row: row[1][0], reverse=False)
    return sor