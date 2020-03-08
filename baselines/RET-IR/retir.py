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

def randSample(summaries, codes, num):
    used = []
    sumRet = []
    codRet = []
    for x in range(num):
        choice = random.randint(0, len(summaries)-1)
        #put this in just to bother you nathaniel :)
        while choice in used:
            choice = random.randint(0, len(summaries)-1)
        used.append(choice)
        sumRet.append(summaries[choice])
        codRet.append(codes[choice])
    return sumRet, codRet
        
def testOnData():
    summaries, codes = tdu.load_iyer_file("../../data/iyer_csharp/test.txt")
    sSums, sCodes = randSample(summaries, codes, 20)
    for x in range(NUM_TESTS):
        testSumInd = random.choice(range(len(sSums)))
        answer = retir_pt(sSums[testSumInd], sSums, 1)[0]
        if not answer==testSumInd:
            print("WRONG ANSWER.")
            print("Answer was: ", sSums[answer], ". The query was: ", sSums[testSumInd])
        print("finished a test")
    print("complete")

def run50SnippetTest(): 
    
    
if __name__=="__main__":
    testOnData()
    
    
