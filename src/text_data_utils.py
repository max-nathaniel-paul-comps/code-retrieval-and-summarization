import re
import numpy as np
import gensim
import csv
import html
from typing import Tuple, List
from tensorflow.keras.preprocessing.sequence import pad_sequences


def remove_excess_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r'(\s\s+)', ' ', text)
    return text


def preprocess_language(language: str) -> str:
    language = language.replace('\n', ' ')
    language = html.unescape(language)
    language = remove_excess_whitespace(language)
    if language[-1] == '?':
        language = language[:-1]
    for opener in ['how do i ', 'how do you ', 'how can i ', 'how to ', 'best way to ', 'can i ',
                   'is there a way to ', 'easiest way to ', 'best implementation for ',
                   'best implementation of ', 'what is the best way to ', 'what is the proper way to ']:
        if language.lower().startswith(opener):
            language = language[len(opener):]
    if not language.startswith("<s>"):
        language = "<s>" + language + "</s>"
    return language


def batch_proc(lis, fun):
    return [fun(li) for li in lis]

128
def preprocess_source_code(source_code: str) -> str:
    source_code = re.sub(r'(?<![:\"])(//.*?\n)', ' ', source_code)
    source_code = source_code.replace('\n', ' ')
    source_code = html.unescape(source_code)
    source_code = remove_excess_whitespace(source_code)
    if not source_code.startswith("<s>"):
        source_code = "<s>" + source_code + "</s>"
    return source_code


def tokenize_text(text: str) -> List[str]:
    words_re = re.compile(r'(\w+|[^\w\s])')
    return ['<s>'] + words_re.findall(text) + ['</s>']


def trim_to_len(summaries, codes, max_summary_len, max_source_code_len):
    assert len(summaries) == len(codes)
    trimmed_summaries = []
    trimmed_codes = []
    for i in range(len(summaries)):
        if len(summaries[i]) <= max_summary_len and len(codes[i]) <= max_source_code_len:
            trimmed_summaries.append(summaries[i])
            trimmed_codes.append(codes[i])
    trimmed_summaries = pad_sequences(trimmed_summaries, maxlen=max_summary_len, padding='post', value=0)
    trimmed_codes = pad_sequences(trimmed_codes, maxlen=max_source_code_len, padding='post', value=0)
    return trimmed_summaries, trimmed_codes


def load_edinburgh_dataset(path: str):
    train_summaries = batch_proc(open(path + "/data_ps.descriptions.train.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    train_codes = batch_proc(open(path + "/data_ps.bodies.train.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    val_summaries = batch_proc(open(path + "/data_ps.descriptions.valid.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    val_codes = batch_proc(open(path + "/data_ps.bodies.valid.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    test_summaries = batch_proc(open(path + "/data_ps.descriptions.test.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    test_codes = batch_proc(open(path + "/data_ps.bodies.test.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    return train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes


def load_csv_dataset(csv_filename: str):
    file = open(csv_filename, encoding='UTF8')
    reader = csv.reader(file)
    summaries = []
    codes = []
    reader.__next__()
    for row in reader:
        summary = row[0]
        summary = preprocess_language(summary)
        code = row[1]
        code = preprocess_source_code(code)
        summaries.append(summary)
        codes.append(code)
    assert len(summaries) == len(codes)
    val_point = int(len(summaries) * 0.8)
    test_point = int(len(summaries) * 0.9)
    train_summaries = summaries[:val_point]
    train_codes = codes[:val_point]
    val_summaries = summaries[val_point:test_point]
    val_codes = codes[val_point:test_point]
    test_summaries = summaries[test_point:]
    test_codes = codes[test_point:]
    return train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes


def load_iyer_file(filename: str) -> Tuple[List[str], List[str]]:
    file_contents = open(filename).readlines()
    summaries = []
    codes = []
    for line in file_contents:
        items = line.split('\t')
        if len(items) == 5:
            split_line = line.split('\t')
            summary = preprocess_language(split_line[2])
            code = preprocess_source_code(split_line[3])
            summaries.append(summary)
            codes.append(code)
    return summaries, codes


def tokenize_texts(texts: List[str]) -> List[List[str]]:
    tokenized = []
    for text in texts:
        tokenized.append(tokenize_text(text))
    return tokenized


def subword_encode(summary_tokenizer, code_tokenizer, max_summary_len, max_code_len,
                   train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes):

    train_summaries = [summary_tokenizer.encode(summary) for summary in train_summaries]
    train_codes = [code_tokenizer.encode(code) for code in train_codes]
    val_summaries = [summary_tokenizer.encode(summary) for summary in val_summaries]
    val_codes = [code_tokenizer.encode(code) for code in val_codes]
    test_summaries = [summary_tokenizer.encode(summary) for summary in test_summaries]
    test_codes = [code_tokenizer.encode(code) for code in test_codes]

    train_summaries, train_codes = trim_to_len(train_summaries, train_codes, max_summary_len, max_code_len)
    val_summaries, val_codes = trim_to_len(val_summaries, val_codes, max_summary_len, max_code_len)
    test_summaries, test_codes = trim_to_len(test_summaries, test_codes, max_summary_len, max_code_len)

    return train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes


def main():
    ex_dataset_file = open("../data/iyer_csharp/train.txt").readlines()
    ex_dataset = []
    for line in ex_dataset_file:
        items = line.split('\t')
        if len(items) == 5:
            ex_dataset.append(line.split('\t')[2])

    print(ex_dataset[0])

    tokenized = tokenize_texts(ex_dataset)
    print(tokenized[0])


if __name__ == "__main__":
    main()
