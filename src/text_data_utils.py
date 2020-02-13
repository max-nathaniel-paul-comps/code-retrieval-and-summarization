import re
import numpy as np
import gensim


def tokenize_texts(texts: list) -> list:
    words_re = re.compile(r'(\w+|[,./?<>!@#$%^&*()_\-+=`~{}|\[\]\\:;\'"])')
    tokenized = []
    for text in texts:
        tokenized.append(['<s>'] + words_re.findall(text) + ['</s>'])
    return tokenized


def tokenized_texts_to_tensor(tokenized: list, wv: gensim.models.KeyedVectors, max_len: int) -> np.ndarray:
    assert max_len >= max(len(text) for text in tokenized)
    tensor = np.zeros((len(tokenized), max_len, wv.vector_size), dtype=np.float32)
    for i in range(len(tokenized)):
        for j in range(len(tokenized[i])):
            if tokenized[i][j] in wv:
                tensor[i][j] = wv[tokenized[i][j]]
            else:
                tensor[i][j] = np.zeros((wv.vector_size,))
    return tensor


def tensor_to_tokenized_texts(tensor: np.ndarray, wv: gensim.models.KeyedVectors) -> list:
    texts = []
    for tokens_tensor in tensor:
        text = []
        for i in range(0, len(tokens_tensor)):
            similar = wv.similar_by_vector(tokens_tensor[i])[0]
            if similar[1] > 0.5:
                text.append(similar[0])
            else:
                text.append("<UNK>")
            if text[-1] == "</s>":
                break
        texts.append(text)
    return texts


def main():
    ex_dataset_file = open("../data/iyer/train.txt").readlines()
    ex_dataset = []
    for line in ex_dataset_file:
        items = line.split('\t')
        if len(items) == 5:
            ex_dataset.append(line.split('\t')[2])

    print(ex_dataset[0])

    tokenized = tokenize_texts(ex_dataset)
    print(tokenized[0])

    wv = gensim.models.Word2Vec(tokenized).wv

    max_len = max(len(text) for text in tokenized)
    tensor = tokenized_texts_to_tensor(tokenized, wv, max_len)
    print(tensor.shape)

    print(tensor_to_tokenized_texts(np.array([tensor[0]]), wv)[0])


if __name__ == "__main__":
    main()
