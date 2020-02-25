import cosineSim as cs
import tfidf as tf
import sys
import random
sys.path.append('../../src')
import text_data_utils as tdu

TF_SCHEME = 4
NUM_TESTS = 10
def topXInds(lis, x):
    indices = []
    for i in range(x):
        indices.append(lis.index(max(lis)))
        lis[lis.index(max(lis))] = -1
    return indices

def retir(query, candidates, numToReturn):
    #RETURNS INDICES OF X HIGHEST CANDIDATES, where X is numToReturn.
    qTokens = tdu.tokenize_text(query)
    qScores = []
    cTokens = tdu.tokenize_texts(candidates)
    cScoreLists = []
    for token in qTokens:
        qScores.append(tf.tfidf(token, query, candidates, TF_SCHEME))
    for cTokenList in range(len(cTokens)):
        temp = []
        for token in cTokens[cTokenList]:
            temp.append(tf.tfidf(token, candidates[cTokenList], candidates, TF_SCHEME))
        cScoreLists.append(temp)

    similarities = []
    for cS in cScoreLists:
        if len(cS) > len(qScores):
            similarities.append(cs.cosSim(qScores, cS[0:len(qScores)]))
        elif len(cS) < len(qScores):
            similarities.append(cs.cosSim(qScores[0:len(cS)], cS))
        else:
            similarities.append(cs.cosSim(qScores, cS))
    return topXInds(similarities, numToReturn)

def retir_pt(query, candidates, numToReturn):
    #pt=pre-tokenized
    #RETURNS INDICES OF X HIGHEST CANDIDATES, where X is numToReturn.
    qTokens = query
    qScores = []
    cTokens = candidates
    cScoreLists = []
    for token in qTokens:
        qScores.append(tf.tfidf_pt(token, query, candidates, TF_SCHEME))
    for cTokenList in range(len(cTokens)):
        temp = []
        for token in cTokens[cTokenList]:
            temp.append(tf.tfidf_pt(token, candidates[cTokenList], candidates, TF_SCHEME))
        cScoreLists.append(temp)

    similarities = []
    for cS in cScoreLists:
        if len(cS) > len(qScores):
            similarities.append(cs.cosSim(qScores, cS[0:len(qScores)]))
        elif len(cS) < len(qScores):
            similarities.append(cs.cosSim(qScores[0:len(cS)], cS))
        else:
            similarities.append(cs.cosSim(qScores, cS))
    return topXInds(similarities, numToReturn)

def testOnData():
    summaries, codes = tdu.load_iyer_file("../../data/iyer/test.txt")
    for x in range(NUM_TESTS):
        testSumInd = random.choice(range(len(summaries)))
        answer = retir_pt(summaries[testSumInd], summaries, 1)[0]
        if not answer==summaries[testSumInd]:
            print("WRONG ANSWER.")
        print("finished a test")
    print("complete")
    
if __name__=="__main__":
    testOnData()
    
    
