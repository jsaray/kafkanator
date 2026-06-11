
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import Counter
from kafkanator.util import transform_dict_keys_to_str,default_row_highlighting
import numpy as np

def statistical_parity_data(df,sensitive_attribute,predict_column,reality_column):
    """This method computes a table that summarizes predictions over sensitive attributes.

    Args:
        df (pandas DataFrame): dataframe . It must contain one sensitive attribute column S containing {s1,s2,s3,...,sn} different sensitive attributes values and one prediction column P containing different categorical predictions {p1,p2,..,pn }.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str):  The column for predictions.
    
    Returns:
        DataFrame : a dataframe whose rows are : < s1 , p1 , nb of predictions p1 > , < s1 , p2, nb of predictions p2 >  , ...
    """
    df = df.drop(df[ df[ predict_column ].isnull()].index)
    vals_sens_attribute = set(df[ sensitive_attribute ].values)
    vals_predictions = set(df[ predict_column ].values)
    bpdata = []
    pred_order = []
    for v in vals_sens_attribute : 
        for p in vals_predictions:
            num = df[(df[sensitive_attribute] == v) & (df[predict_column] == p)].shape
            bpdata.append({'attr': v, 'prediction':p,'number': num[0] })
            barplot_data = pd.DataFrame(data=bpdata,columns=['attr','prediction','number'])
    return barplot_data

def statistical_parity(df,sensitive_attribute,predict_column,reality_column):
    """This method computes statistical parity on values of a specified sensitive attribute.
    Args:
        df (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
        containing {v1,v2,v3,...,vn} different sensitive attributes values, and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        {v1: p1, v2: p2} : a dictionary containing the P1 class per value in set of columns S  .
    """
    groups = df.groupby(by=sensitive_attribute)
    stock = {}
    for (k,v) in groups :
        totpos = sum(v[predict_column])
        totalrows = v.shape[0]
        stock[k] =  totpos / totalrows
    return stock

def disparate_impact (df,sensitive_attribute,predict_column,reality_column):
    """This method computes disparate impact on values of a specified sensitive attribute.
    Args:
        df (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
        containing {v1,v2,v3,...,vn} different sensitive attributes values, and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        {v1: prev1, v2: prev2} : a dictionary containing the prevalence per value in set of columns S  .
    """
    groups = df.groupby(by=sensitive_attribute)
    stock = {}
    for (k,v) in groups :
        cm = confusion_matrix( v[reality_column] ,  v[predict_column] )
        DI = (cm[1][1]  +  cm[1][0])/ (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        stock[k] =  DI
    return stock

def equal_opportunity(df,sensitive_attribute,predict_column,reality_column):
    """This method computes disparate impact on values of a specified sensitive attribute.
    Args:
        df (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
        containing {v1,v2,v3,...,vn} different sensitive attributes values, and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        {v1: tpr1, v2: tpr2} : a dictionary containing the TPR per value in set of columns S  .
    """
    dictret_tpr = {}
    groups = df.groupby(by=sensitive_attribute)
    for (k,v) in groups :
        cm = confusion_matrix( v[reality_column] ,  v[predict_column] )
        TPR = cm[1][1] / (cm[1][0] + cm[1][1])
        dictret_tpr[k] = TPR
    return dictret_tpr

def equalized_odds(df,sensitive_attribute,predict_column,reality_column):
    """This method computes disparate impact on values of a specified sensitive attribute.
    Args:
        df (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
        containing {v1,v2,v3,...,vn} different sensitive attributes values, and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        {v1: fpr1,tpr1, v2: fpr2,tpr2} : a dictionary containing the FPR,TPR per value in set of columns S  .
    """
    ekodd = {}
    groups = df.groupby(by=sensitive_attribute)
    for (k,v) in groups :
        cm = confusion_matrix( v[reality_column] ,  v[predict_column] )
        FPR = cm[0][1] / (cm[0][1] + cm[0][0])
        TPR = cm[1][1] / (cm[1][0] + cm[1][1])
        ekodd[k] = str(FPR) + ',' + str(TPR)
    return ekodd

def predictive_parity(df,sensitive_attribute,predict_column,reality_column):
    """This method computes disparate impact on values of a specified sensitive attribute.
    Args:
        df (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
        containing {v1,v2,v3,...,vn} different sensitive attributes values, and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        {v1: ppv1, v2: ppv2} : a dictionary containing the FPR,TPR per value in set of columns S  .
    """
    dictret_ppv = {}
    groups = df.groupby(by=sensitive_attribute)
    for (k,v) in groups :
        cm = confusion_matrix( v[reality_column] ,  v[predict_column] )
        ppv = cm[1][1] / (cm[1][1] + cm[0][1])
        dictret_ppv[k] = ppv
    return dictret_ppv

def fpr_fnr(df,sensitive_attribute,predict_column,reality_column):
    """This method computes false positive rate and false negative rate on subpopulations ( see <>HERE<> ).

    Args:
        df (pandas DataFrame): dataframe . It must contain one sensitive attribute column S containing {s1,s2,s3,...,sn} different sensitive attributes values and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        (fpr,fnr) : a tuple containing in position 0 the false positive rate, and in position 1 the false negative rate.
    """
    dictret_fpr = {}
    dictret_fnr = {}
    
    for x in list(set(df[sensitive_attribute].values)):
        dictret_fpr[x] = ''
        dictret_fnr[x] = ''
    groups = df.groupby(by=[sensitive_attribute])
    
    for (k,v) in groups :
        cm = confusion_matrix( v[reality_column] ,  v[predict_column] )
        FPR = cm[0][1] / (cm[0][1] + cm[0][0])
        FNR = cm[1][0] / ( cm[1][0] + cm[1][1])
        dictret_fpr[k[0]] = FPR
        dictret_fnr[k[0]] = FNR
    return dictret_fpr, dictret_fnr

def build_last_column(df,label_last_column):
    """This PRIVATE method compute last column of summarized fairness measure table.
    Args:
        df (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
    Returns:
        list : a list with last column
    """
    column = []
    for i in range(0,df.shape[0]):
        ind = df.index[i]
        if (ind == 'DEMOGRAPHIC PARITY - P1') or (ind == 'EQUAL OPPORTUNITY - TPR') or (ind == 'PREDICTIVE PARITY - PPV'):
            print ( ' first if ', str(len(df.columns)))
            if len(df.columns) == 2:
                d = abs(df.iloc[i,0] - df.iloc[i,1])
                column.append(d)
            elif len(df.columns) > 2:
                d = max(df.iloc[i]) - min(df.iloc[i])
                column.append(d)
        elif ind == 'EQUALIZED ODDS - (TPR,FPR)':
            print ( ' second if ')
            tu1 = [float(x) for x in df.iloc[i,0].split(',')]
            tu2 =[float(x) for x in df.iloc[i,1].split(',')]
            tu = np.subtract ( tu1 ,tu2) 
            tupled = tuple(np.absolute(tu))
            column.append(','.join([str(x) for x in tupled]))
        elif ind == 'DISPARATE IMPACT - PREVALENCE':
            print ( ' third if ')
            d = (df.iloc[i,0] /  df.iloc[i,1]) if df.iloc[i,0] < df.iloc[i,1] else  (df.iloc[i,1] / df.iloc[i,0])
            column.append(d)
    print ( 'retriv column with len ',len(column))
    return column

def fairness_metrics_table(dataset,sensitive_attribute,predict_column,reality_column,aggregate_metrics=False,function_last_column=None,label_last_column='DELTA'):
    """This method compute a fairness measures summary table.
    Args:
        dataset (pandas DataFrame): dataframe . It must contain one or more sensitive attribute columns S 
        containing {v1,v2,v3,...,vn} different sensitive attributes values, and one prediction column P containing a binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
        predict_column (str): the column where dataframe df stores prediction.
        reality_column (str): the column where dataframe df stores what happens in reality.
    Returns:
        DataFrame : a dataframe summarizing fairness measures .
    """
    colormap = []
    indices = ['DEMOGRAPHIC PARITY - P1','EQUAL OPPORTUNITY - TPR','PREDICTIVE PARITY - PPV','DISPARATE IMPACT - PREVALENCE']
    (sp,eo,pp,eodd,di) = (statistical_parity(dataset,sensitive_attribute,predict_column,reality_column) , 
    equal_opportunity(dataset,sensitive_attribute,predict_column,reality_column),
    predictive_parity(dataset,sensitive_attribute,predict_column,reality_column),
    equalized_odds(dataset,sensitive_attribute,predict_column,reality_column),
    disparate_impact(dataset,sensitive_attribute,predict_column,reality_column))
    sp_strkeys = transform_dict_keys_to_str(sp)
    print ('ks ', sp_strkeys)
    lcols = list(sp_strkeys.keys())
    print ( ' lcosl ', lcols)
    df = pd.DataFrame(index=indices,columns=lcols)
    print ('sta party output ',sp)
    for (k,v) in sp_strkeys.items():
        df.loc['DEMOGRAPHIC PARITY - P1',k] = sp_strkeys[k]
    eo_strkeys = transform_dict_keys_to_str(eo)
    for (k,v) in eo_strkeys.items():
        df.loc['EQUAL OPPORTUNITY - TPR',k] = eo_strkeys[k]
    pp_strkeys = transform_dict_keys_to_str(pp)
    for (k,v) in pp_strkeys.items():
        df.loc['PREDICTIVE PARITY - PPV',k] = pp_strkeys[k]
    di_strkeys = transform_dict_keys_to_str(di)
    for (k,v) in di_strkeys.items():
        df.loc['DISPARATE IMPACT - PREVALENCE',k] = di_strkeys[k]
    colormap = []
    if aggregate_metrics == True :
        assert(label_last_column!=None)
        if function_last_column != None : 
            df[label_last_column] =  function_last_column 
        else:
            df[label_last_column] = build_last_column(df,label_last_column)
    return df