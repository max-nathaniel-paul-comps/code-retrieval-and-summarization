import re
import numpy as np
import gensim
from typing import Tuple, List


def tokenize_text(text: str) -> List[str]:
    words_re = re.compile(r'(\w+|[,./?<>!@#$%^&*()_\-+=`~{}|\[\]\\:;\'"])')
    return ['<s>'] + words_re.findall(text) + ['</s>']


def load_iyer_file(filename: str, max_len: int = 0) -> Tuple[List[List[str]], List[List[str]]]:
    file_contents = open(filename).readlines()
    summaries = []
    codes = []
    for line in file_contents:
        items = line.split('\t')
        if len(items) == 5:
            split_line = line.split('\t')
            summary = tokenize_text(split_line[2].lower())
            code = tokenize_text(split_line[3])
            if max_len == 0 or (len(summary) < max_len and len(code) < max_len):
                summaries.append(summary)
                codes.append(code)
    return summaries, codes


def tokenize_texts(texts: List[str]) -> List[List[str]]:
    tokenized = []
    for text in texts:
        tokenized.append(tokenize_text(text))
    return tokenized


def tokenized_texts_to_tensor(tokenized: List[List[str]], wv: gensim.models.KeyedVectors, max_len: int) -> np.ndarray:
    assert max_len >= max(len(text) for text in tokenized)
    tensor = np.zeros((len(tokenized), max_len, wv.vector_size), dtype=np.float32)
    for i in range(len(tokenized)):
        for j in range(len(tokenized[i])):
            if tokenized[i][j] in wv:
                tensor[i][j] = wv[tokenized[i][j]]
    return np.array(tensor)


def tensor_to_tokenized_texts(tensor: np.ndarray, wv: gensim.models.KeyedVectors) -> List[List[str]]:
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
