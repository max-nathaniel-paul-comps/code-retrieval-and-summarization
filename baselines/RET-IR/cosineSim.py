import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Code from https://skipperkongen.dk/2018/09/19/cosine-similarity-in-python/
def cosSim(a, b):
    #ASSUME a and b are python lists
    a = np.array(a)
    b = np.array(b)
 
    # manually compute cosine similarity
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
     
    # use library, operates on sets of vectors
    aa = a.reshape(1,len(a))
    ba = b.reshape(1,len(b))
    cos_lib = cosine_similarity(aa, ba)
     
    print(
        dot,
        norma,
        normb,
        cos,
        cos_lib[0][0]
    )
    return cos_lib

if __name__=="__main__":
    #cosSim()
    print("nothing yet")
