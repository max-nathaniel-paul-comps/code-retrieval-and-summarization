import math
import re
import sys
sys.path.append('../../src')
import text_data_utils as tdu
    

def termFrequency(term, doc_pt, scheme, needsTokenizing=False):
    #pt=pre-tokenized
    #Simple count
    if needsTokenizing:
        doc_pt = tdu.tokenize_texts(doc_pt)
    termCount = countInDoc(term, doc_pt)
    if scheme==0:
        return termCount
    #Boolean frequency
    elif scheme==1:
        if termCount > 0:
            return 1
        else:
            return 0
    #Adjusted for document length
    elif scheme==2:
        return termCount/len(doc_pt)
    #logarithmically scaled freq
    elif scheme==3:
        return math.log(1 + termCount)
    #augmented frequency, raw freq divided by freq of most occurring term
    elif scheme==4:
        return 0.5 + (0.5 * (termCount/(countHighestFreq(doc_pt))))
    else:
        print("ERROR: Invalid scheme")
        return None
        

def countInDoc(word, docWords):
    tCount = 0
    for w in docWords:
        if w == word:
            tCount += 1
    return tCount

def countHighestFreq(docWords):
    maxCount = 0
    for w in docWords:
        maxCount = max(maxCount, countInDoc(w, docWords))
    return maxCount

def inverseDocFrequency(term, docs):
    N = len(docs)
    denom = 1
    for d in docs:
        spl = tdu.tokenize_text(d)
        if countInDoc(term, spl) > 0:
            denom += 1
    return 1.0 + math.log(float(N/denom))

def inverseDocFrequency_pt(term, docs, idf_scheme=0):
    N = len(docs)
    occurrences = 1
    for d in docs:
        if countInDoc(term, d) > 0:
            occurrences += 1

    if idf_scheme == 0:
        return 1.0 + math.log(float(N/occurrences))
    elif idf_scheme == 1:
        return math.log((N-occurrences)/occurrences)

def tfidf(term, doc, docs, scheme):
    return termFrequency(term, doc, scheme, True) * inverseDocFrequency(term, docs)

def tfidf_pt(term, doc, docs, scheme):
    #so this assumes that doc and each d in docs area already token lists
    return termFrequency(term, doc, scheme) * inverseDocFrequency_pt(term, docs)

if __name__=="__main__":
    print("TFIDF test stuff running")
    s = "here's a doc with word in it"
    a = "here's this other doc without that in it"
    b = "here's my ass, isn't it lovely. word."
    c = "a sentence entirely without that thing"
    docs = [s,a,b,c]
    print("term freq: " + str(termFrequency("word", s, 0)))
    print("idf: " + str(inverseDocFrequency("word", docs)))
    print("tfidf: " + str(tfidf("word", s, docs, 0)))
