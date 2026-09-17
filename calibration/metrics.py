def emitKey(x,n_bins):
    # x is a pair (prediction,real), we need both because we need to remember the number of 1 in the slot.
    bin_length = 1 / n_bins
    slot = (x[0]*100) / (bin_length*100)
    if x[0]==1:
        slot = n_bins-1
    #print ( ' slot of ', x[0] , ' is ' , int(slot))
    return (int(slot),x)


def ece(pred_test_zip, n_bins=10):
    bins = list ( map( lambda x : emitKey(x,n_bins) , pred_test_zip ) )
    #bins will be of the form [(slot1, (p1,r1), slot2 (p2,r2),(p3,r3), (p4,r4), (p5,r5) ]
    org_hash = {}
    for b in bins: 
        if org_hash.get( b[0] ) == None : 
            org_hash[ b[0] ] = []
            org_hash[ b[0] ].append(b[1])
        else:
            org_hash[ b[0] ].append(b[1])
    total_ece = 0
    for (k,v) in org_hash.items():
        weight = len(v) / len(pred_test_zip)
        list_for_conf= [x[0] for x in v ]
        list_for_acc = [x[1] for x in v ]
        conf = sum(list_for_conf)/len(list_for_conf)
        acc = sum ( list_for_acc ) / len ( list_for_acc )
        total_ece = total_ece + (weight * abs(acc - conf) )
    return total_ece