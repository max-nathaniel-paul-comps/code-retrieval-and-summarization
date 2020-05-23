import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Code from https://skipperkongen.dk/2018/09/19/cosine-similarity-in-python/
def cosSim(a, b):
    #ASSUME a and b are python lists
    if len(a) < len(b):
        b = b[:len(a)]
    else:
        a = a[:len(b)]
    if not len(a) == len(b):
        print("cosSim on vectors of non-equal lengths")
    a = np.array(a)
    b = np.array(b)
     
    # use library, operates on sets of vectors
    aa = a.reshape(1,len(a))
    ba = b.reshape(1,len(b))
    if (not aa.shape == (1, 0)) and (not ba.shape == (1, 0)):
        cos_lib = cosine_similarity(aa, ba)
        return cos_lib[0][0]
    else:
        return 0

if __name__=="__main__":
    a = [0.5, 1.7, 2.0]
    b = [1000, 1.6, 3.0]
    print(cosSim(a, b))
