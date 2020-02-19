import gensim

gensim.scripts.glove2word2vec.glove2word2vec('summaries_vectors.txt', 'w2v_format_summaries_vectors.txt')
gensim.scripts.glove2word2vec.glove2word2vec('codes_vectors.txt', 'w2v_format_codes_vectors.txt')