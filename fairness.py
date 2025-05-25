
import pandas as pd

def statistical_parity_data(df,sensitive_attribute,predict_column):
	"""This method computes a table that summarizes predictions over sensitive attributes.

    Args:
        df (pandas DataFrame): dataframe . It must contain one sensitive attribute column S containing {s1,s2,s3,...,sn} different sensitive attributes values and one prediction column P containing different categorical
        predictions {p1,p2,..,pn }.
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

def fpr_fnr(df,sensitive_attribute):
	"""This method computes false positive rate and false negative rate on subpopulations ( see <>HERE<> ).

    Args:
        df (pandas DataFrame): dataframe . It must contain one sensitive attribute column S containing {s1,s2,s3,...,sn} different sensitive attributes values and one prediction column P containing a
        binary prediction {1,0}.
        sensitive_attribute (str): The column designing the sensitive attribute : sex, age, handicap, nationality etc.
    
    Returns:
        (fpr,fnr) : a tuple containing in position 0 the false positive rate, and in position 1 the false negative rate.
    """
    dictret_fpr = {}
    dictret_fnr = {}
    
    for x in list(set(df[sensitive_attribute].values)):
        dictret_fpr[x] = ''
        dictret_fnr[x] = ''
    groups = df.groupby(by=[sensitive_attribute])
    
    print ( ' Dictionnary FP ', dictret_fpr, ' FN ',dictret_fnr)
    for (k,v) in groups :
        print ( ' heigh of v ', v.shape[0], ' k is ', k)
        cm = confusion_matrix( v['default'] ,  v['Prediction'] )
        FPR = cm[0][1] / (cm[0][1] + cm[0][0])
        FNR = cm[1][0] / ( cm[1][0] + cm[1][1])
        dictret_fpr[k[0]] = FPR
        dictret_fnr[k[0]] = FNR
    return dictret_fpr, dictret_fnr