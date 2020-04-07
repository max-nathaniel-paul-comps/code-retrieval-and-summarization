import retir as rt
import numpy as np
def mrr(results):
    #input should be a list of the ranks of the correct query for each
    #50-snippet test.
    #rQ = 1/len(results)
    #rankSum = 0
    rankArr = []
    for r in results:
        rankArr.append(1/(r+1))
        #rankSum += 1/(r+1)
    return np.mean(rankArr)
    
    
