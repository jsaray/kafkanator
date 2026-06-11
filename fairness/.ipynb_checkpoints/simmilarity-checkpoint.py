
def simmilarity_fairness_hash( data, sensitive_column, sensitive_attribute_values ,simmilarity_attr_hsh ,numrows,simmilarity_distance='catnum_simmilarity_distance'  ):
    wo = data[ data[sensitive_column] == sensitive_attribute_values[0]].reset_index()
    wo = wo.drop(['index'],axis=1)
    ma = data[ data[sensitive_column] == sensitive_attribute_values[1]].reset_index()
    ma = ma.drop(['index'],axis=1)
    # For the moment, only 2 values in sensitive_attribute_values allowed. if more than 2 we have to do like the correlation plots
    hDict = {}
    for i,w in wo.iloc[0:numrows].iterrows():
        if i % 100 == 0:
            print ( 'Analyzing row ' , i)
        for j,h in ma.iloc[0:numrows].iterrows():
            if simmilarity_distance == 'catnum_simmilarity_distance':
                reto = categorical_simmilarity_distance(w,h,simmilarity_attr_hsh)
            #print ( ' i ', i,' j ', j , ' reto ', reto ,  ' fnlwgt W ' , w['fnlwgt'] , ' fnlwgt H ' , h['fnlwgt'] )
            d = sum(reto[0].values())
            hDict[(i,j)] = (d,reto[1])
    
    sor = sorted(hDict.items(), key=lambda row: row[1][0], reverse=False)
    return sor