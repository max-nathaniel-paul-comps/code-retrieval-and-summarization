import textdistance
import random
import sys
from operator import itemgetter
from nltk import download
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
sys.path.append('../../src')
import text_data_utils as tdu

def setup(numToTake):
    summaries, codes = tdu.load_iyer_file("../../data/iyer_csharp/test.txt")
    sumSnips, codeSnips = randSample(summaries, codes, numToTake)
    return sumSnips, codeSnips

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

def shuffleQuery(q):
    q = q[3:]
    q = q[:-4]
    returner = ""
    for w in q.split(" "):
        if random.randint(0, 9) > 2:
            returner += w + " "
    returner = "<s>" + returner + "</s>"
    return returner

def ir(snippet, sums, codes):
    assert len(sums)==len(codes)
    dists = []
    for c in range(len(codes)):
        dists.append([textdistance.levenshtein(snippet, codes[c]), c])
    sorted(dists, key=itemgetter(0))
    return sums[dists[0][1]]

def unTokenize(tokenListList):
    returner = []
    for tokenList in tokenListList:
        temp = ""
        for t in tokenList:
            temp += t + " "
        returner.append(temp)
    return returner

def eval_ir(sums, codes, codeSnippet, codeSnippetIndex, areTokenized):
    correct = sums[codeSnippetIndex]
    candidate = ir(shuffleQuery(codeSnippet), sums, codes)
    bleu = -1.0
    meteor = -1.0
    if areTokenized:
        bleu = sentence_bleu(sums, candidate)
        meteor = round(meteor_score(unTokenize(sums), candidate), 6)
    else:
        print("BLEU NON-TOKENIZED NOT IMPLEMENTED")
        meteor = round(meteor_score(sums, candidate), 6)
    print("METEOR SCORE: ", meteor)
    print("BLEU-4 SCORE: ", bleu)
    return meteor, bleu

def test_ir(sums, codes, num_tests, intensify, test_num):
    assert len(sums) == len(codes)
    msum = 0
    bsum = 0
    for x in range(num_tests):
        if intensify:
            sums, codes = setup(test_num)
        cs = random.choice(list(range(len(codes))))
        tmeteor, tbleu = eval_ir(sums, codes, codes[cs], cs, True)
        msum += tmeteor
        bsum += tbleu
    msum = msum / num_tests
    bsum = bsum / num_tests
    print("Average METEOR: ", msum)
    print("Average BLEU: ", bsum)

def test_ir_iyer(num_tests, test_num, intensify):
    sumSnips, codeSnips = setup(test_num)
    #sums, codes = tdu.trim_to_len(summaries, codess, trim_num, trim_num)
    test_ir(sumSnips, codeSnips, num_tests, intensify, test_num)

if __name__=="__main__":
    #download('wordnet')
    print("How many tests would you like to run? (10)")
    num_tests = int(input())
    print("How many snippets in the test? (50)")
    num_snips = int(input())
    print("Intensify? y/n")
    do = False
    if input()=="y":
        do = True
    test_ir_iyer(num_tests, num_snips, do)
        
    


