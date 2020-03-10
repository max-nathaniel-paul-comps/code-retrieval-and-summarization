import textdistance
import random
import sys
from operator import itemgetter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
sys.path.append('../../src')
import text_data_utils as tdu

def ir(snippet, sums, codes):
    assert len(sums)==len(codes)
    dists = []
    for c in range(len(codes)):
        dists.append([textdistance.levenshtein(snippet, codes[c]), c])
    sorted(dists, key=itemgetter(0))
    return sums[dists[0][1]]

def eval_ir(sums, codes, codeSnippet, codeSnippetIndex, areTokenized):
    correct = sums[codeSnippetIndex]
    candidate = ir(codeSnippet, sums, codes)
    bleu = -1.0
    meteor = -1.0
    if areTokenized:
        bleu = sentence_bleu(sums, candidate)
        print("METEOR TOKENIZED NOT IMPLEMENTED")
    else:
        print("BLEU NON-TOKENIZED NOT IMPLEMENTED")
        meteor = round(meteor_score(sums, candidate), 6)
    print("METEOR SCORE: ", meteor)
    print("BLEU-4 SCORE: ", bleu)

def test_ir(sums, codes, num_tests):
    assert len(sums) == len(codes)
    for x in range(num_tests):
        cs = random.choice(list(range(len(codes))))
        eval_ir(sums, codes, codes[cs], cs, True)

def test_ir_iyer(num_tests, trim_num, test_num):
    summaries, codess = tdu.load_iyer_file("../../data/iyer_csharp/test.txt")
    #sums, codes = tdu.trim_to_len(summaries, codess, trim_num, trim_num)
    test_ir(summaries[:test_num], codess[:test_num], num_tests)

if __name__=="__main__":
    test_ir_iyer(10, 100, 50)
        
    


