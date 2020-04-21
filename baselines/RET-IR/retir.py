import cosineSim as cs
import tfidf
import evaluator as ev
import sys
import random
import string
sys.path.append('../../src')
import text_data_utils as tdu

DEBUG_PRINT = False
DO_SHIFTING = False
TF_SCHEME = 0
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
    cTokens = tdu.tokenize_texts(candidates)
    return retir_pt(qTokens, cTokens, numToReturn)

def retir_pt(query, candidates, numToReturn):
    #pt=pre-tokenized
    #RETURNS INDICES OF X HIGHEST CANDIDATES, where X is numToReturn.
    qTokens = query
    qScores = []
    cTokens = candidates
    cScoreLists = []
    for token in qTokens:
        qScores.append(tfidf.tfidf_pt(token, query, candidates, TF_SCHEME))
    for cTokenList in range(len(cTokens)):
        temp = []
        for token in cTokens[cTokenList]:
            temp.append(tfidf.tfidf_pt(token, candidates[cTokenList], candidates, TF_SCHEME))
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

def randSample(summaries, codes, num, doingAlts = False, altSums = None):
    used = []
    sumRet = []
    codRet = []
    altRet = []
    for x in range(num):
        choice = random.randint(0, len(summaries)-1)
        #put this in just to bother you nathaniel :)
        while choice in used:
            choice = random.randint(0, len(summaries)-1)
        used.append(choice)
        sumRet.append(summaries[choice])
        codRet.append(codes[choice])
        if doingAlts:
            altRet.append(altSums[choice])
    if doingAlts:
        return sumRet, codRet, altRet
    else:
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

def setup(numToTake):
    summaries, codes = tdu.load_iyer_file("../../data/iyer_csharp/dev.txt")
    sumSnips, codeSnips = randSample(summaries, codes, numToTake)
    return sumSnips, codeSnips

def setupAlt(numToTake):
    dataset = tdu.load_iyer_dataset("../../data/iyer_csharp/dev.txt", "../../data/iyer_csharp/dev_alternate_summaries.txt")
    summaries = [ex[0] for ex in dataset]
    codes = [ex[1] for ex in dataset]
    alt_summaries = [ex[2] for ex in dataset]
    sumSnips, codeSnips, altSnips = randSample(summaries, codes, numToTake, True, alt_summaries)
    return sumSnips, codeSnips, altSnips

def run50SnippetTest():
    sumSnips, codeSnips = setup(50)
    choice = random.choice(range(len(codeSnips)))
    corSum = sumSnips[choice]
    dPrint("Give a human-annotated title for the following code snippet. This will be used as the query.")
    dPrint(codeSnips[choice])
    query = input()
    ranks = retir_pt(query, sumSnips, 50)
    rankCorrect = ranks.index(corSum)
    return ranks, rankCorrect

def runSingleQuery():
    sumSnips, codeSnips = setup(50)
    choice = random.choice(range(len(codeSnips)))
    corSum = sumSnips[choice]
    corCode = codeSnips[choice]
    dPrint("Give a reasonable query for the following code snippet:")
    dPrint(corCode)
    dPrint("With original summary:")
    dPrint(corSum)
    query = input()
    query = "<s>" + query + "</s>"
    returns = retir_pt(query, sumSnips, 20)
    returnval = returns[0]
    returnCode = codeSnips[returnval]
    returnSum = sumSnips[returnval]
    dPrint("Rank of your correct snippet was: ")
    dPrint(returns.index(choice))
    dPrint("The following code was returned, hopefully the same snippet as before")
    dPrint(returnCode)
    dPrint("Taken from given summary: ")
    dPrint(returnSum)

def shuffleQuery(q, chanceToShuffle):
    assert chanceToShuffle < 1.0 and chanceToShuffle >= 0.0
    #q = q[3:]
    #q = q[:-4]
    returner = ""
    moveOver = False
    mover = ""
    for w in q.split(" "):
        if moveOver:
            moveOver = False
            returner += mover + " "
        ran = random.random()
        if ran > chanceToShuffle:
            returner += w + " "
        else:
            if ran < (chanceToShuffle/2) and DO_SHIFTING:
                moveOver = True
                mover = w
    #returner = "<s>" + returner + "</s>"
    return returner

def runShuffleQuery(numSnips, numTimes):
    listRanks = []
    chance = 0.2
    for x in range(numTimes):
        sumSnips, codeSnips = setup(numSnips)
        corInd = random.choice(range(len(codeSnips)))
        corSum = sumSnips[corInd]
        dPrint("Original summary:")
        dPrint(corSum)
        dPrint("For code:")
        dPrint(codeSnips[corInd])
        shuffled = shuffleQuery(corSum, chance)
        #shuffled = randCharString(len(corSum))
        dPrint("Edited query:")
        dPrint(shuffled)
        returns = retir_pt(shuffled, sumSnips, numSnips)
        dPrint("Correct return was at rank:")
        dPrint(returns.index(corInd))
        listRanks.append(returns.index(corInd))
        print("Finished test ", x)
    print("Final MRR for shuffle: ")
    print(ev.mrr(listRanks))

def runAlternateQuery(numSnips, numTests):
    listRanks = []
    for x in range(numTests):
        sumSnips, codeSnips, altSnips = setupAlt(numSnips)
        corInd = random.choice(range(len(codeSnips)))
        corSum = sumSnips[corInd]
        dPrint("Original summary:")
        dPrint(corSum)
        dPrint("For code:")
        dPrint(codeSnips[corInd])
        altSum = altSnips[corInd][0]
        #altSum = randCharString(len(corSum))
        dPrint("Alternate summary:")
        dPrint(altSum)
        returns = retir(altSum, sumSnips, numSnips)
        dPrint("Correct return was at rank:")
        dPrint(returns.index(corInd))
        dPrint("Top result given:")
        dPrint(sumSnips[returns[0]])
        listRanks.append(returns.index(corInd))
        print("Finished test ", x)
    print("Final MRR for alts: ")
    print(ev.mrr(listRanks))
def randCharString(length):
    returner = ""
    checker = string.ascii_letters + " </>"
    for x in range(length):
        returner += random.choice(checker)
    return returner

def dPrint(string):
    if DEBUG_PRINT:
        print(string)
if __name__=="__main__":
    print("Which test of RET-IR would you like to run?")
    print("(0) TestOnData() (DEPRECATED TESTING FUNC)")
    print("(1) Manual query testing")
    print("(2) Snippet tests using shuffle query testing")
    print("(3) Snippet tests using alternate summaries")
    resp = input()

    if resp=="0":
        testOnData()
    elif resp=="1":
        runSingleQuery()
    elif resp=="2":
        print("How many snippets per test? Recommended 50")
        numSnips = int(input())
        print("How many tests to run?")
        numTests = int(input())
        runShuffleQuery(numSnips, numTests)
    else:
        print("How many snippets per test? Recommended 50")
        numSnips = int(input())
        print("How many tests to run?")
        numTests = int(input())
        runAlternateQuery(numSnips, numTests)
        
