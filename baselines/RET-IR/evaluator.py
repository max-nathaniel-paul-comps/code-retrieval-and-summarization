import retir as rt
def mrr(results):
    #input should be a list of the ranks of the correct query for each
    #50-snippet test.
    rQ = 1/len(results)
    rankSum = 0
    for r in results:
        rankSum += 1/r
    return rQ * rankSum
    
def run50snippetTests(sums, codes, numTests):
    
