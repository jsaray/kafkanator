
def score_by_metric(arr,reverse=False):
    return sorted(arr,reverse=reverse)

def score_by_lambda(arr,func,reverse=False):
    return sorted(arr,key=func,reverse=reverse)





