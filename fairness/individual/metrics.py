
def fraction_score(x,y):
    return x / y

def individual_consistence(x,k_neigbords):
    pct_aggreement = sum(k_neigbords)/ len ( k_neigbords )
    return 1 - abs(x - pct_aggreement)

# Here x , y are lists . Instead of d we will have a list of ds of simmilarest k neighbots, instead of D's we will have D range.
# kneighbors {  (i,j) : (  d , (Ti,Tj)    )  }
def counterfactual_token_fairness_gap ( x , x_counterfactual_predictions   ):
    abss = 0
    print ( 'conterfs preds ', x_counterfactual_predictions)
    for i in range(0,len(x_counterfactual_predictions) ):
        abss = abss + abs( x -  x_counterfactual_predictions[i] )
    return abss / len ( x_counterfactual_predictions )

def sensitive_attribute_gap ( x, neigbors , sensitive_attributes):
        filtx = x[sensitive_attributes]
        
        filtk = neigbors.iloc[:,sensitive_attributes]
        #print ( 'x ', filtx,' neigbor ', filtk,'sa ', sensitive_attributes)  
        sumgapk = 0
        totalsumgap = []
        for k in range(0,filtk.shape[0]):
            sumgapk=0
            for s in range(0,filtk.shape[1]):
                #print ( 'k is ', k, ' s is ', s)
                d = 1 if filtk.iloc[k,s] != filtx[s] else 0 
                sumgapk = sumgapk + d
            totalsumgap.append(sumgapk)
        return sum(totalsumgap)