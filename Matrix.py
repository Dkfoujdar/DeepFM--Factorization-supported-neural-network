import numpy as np

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float) #having the array with [[y,...,yp,...,index],[..],...]
    
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))] #shorting increasingly with second column 
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(actual, pred): # it's to see that how close the acutal values are to the predicted ones
    return gini(actual, pred) / gini(actual, actual)