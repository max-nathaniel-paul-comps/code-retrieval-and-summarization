import textdistance
from operator import itemgetter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

def ir(snippet, sums, codes):
    assert len(sums)==len(codes)
    dists = []
    for c in range(len(codes)):
        dists.append([textdistance.levenshtein(snippet, codes[c]), c])
    sorted(dists, key=itemgetter(0))
    return sums[dists[0][1]]

def eval(sums, codes, codeSnippet, codeSnippetIndex, areTokenized):
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
        
    


