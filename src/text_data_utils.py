import re
import csv
import html
from typing import Tuple, List
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from CSharp4Lexer import *


def remove_excess_whitespace(text: str) -> str:
    text = text.strip()
    text = re.sub(r'(\s\s+)', ' ', text)
    return text


def preprocess_language(language: str) -> str:
    language = language.replace('\\n', ' ').replace('\n', ' ')
    language = html.unescape(language)
    language = remove_excess_whitespace(language)
    if language[-1] == '?':
        language = language[:-1]
    for opener in ['how do i ', 'how do you ', 'how can i ', 'how to ', 'best way to ', 'can i ',
                   'is there a way to ', 'easiest way to ', 'best implementation for ',
                   'best implementation of ', 'what is the best way to ', 'what is the proper way to ',
                   'is it possible to ', 'would it be possible to '
                   'how ', 'c# how to ', 'c# how ', 'c# - ', 'c# ']:
        if language.lower().startswith(opener):
            language = language[len(opener):]
    for closer in [' in c#', ' with c#', ' using c#', ' c#']:
        if language.lower().endswith(closer):
            language = language[:-len(closer)]
    language = language.lower()
    return language


def batch_proc(lis, fun):
    return [fun(li) for li in lis]


def preprocess_source_code(source_code: str) -> str:
    source_code = re.sub(r'(?<![:\"])(//.*?\n)', ' ', source_code)
    source_code = source_code.replace('\\n', ' ').replace('\n', ' ')
    source_code = html.unescape(source_code)
    source_code = remove_excess_whitespace(source_code)
    return source_code


def tokenize_text(text: str) -> List[str]:
    words_re = re.compile(r'(\w+|[^\w\s])')
    return ['<s>'] + words_re.findall(text) + ['</s>']


def sequences_to_tensors(summaries, codes, max_summary_len, max_source_code_len,
                         oversize_sequence_behavior='leave_out', dtype='int32'):
    """
    Prepares a set of tokenized summaries and codes for use in a model by padding them to a fixed length.
    By default, oversize examples are left out, but you may also set oversize_sequence_behavior to 'truncate'
    """
    assert len(summaries) == len(codes)
    trimmed_summaries = []
    trimmed_codes = []
    for i in range(len(summaries)):
        if oversize_sequence_behavior == 'truncate' \
                or len(summaries[i]) <= max_summary_len and len(codes[i]) <= max_source_code_len:
            trimmed_summaries.append(summaries[i])
            trimmed_codes.append(codes[i])
    if len(trimmed_summaries) < len(summaries):
        print("Warning: %s examples were left out because they were oversize" %
              (len(summaries) - len(trimmed_summaries)))
    trimmed_summaries = pad_sequences(trimmed_summaries, maxlen=max_summary_len, padding='post', value=0, dtype=dtype)
    trimmed_codes = pad_sequences(trimmed_codes, maxlen=max_source_code_len, padding='post', value=0, dtype=dtype)
    summaries_tensor = tf.convert_to_tensor(trimmed_summaries)
    codes_tensor = tf.convert_to_tensor(trimmed_codes)
    return summaries_tensor, codes_tensor


def load_edinburgh_dataset(path: str):
    train_summaries = batch_proc(open(path + "/data_ps.descriptions.train.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    train_codes = batch_proc(open(path + "/data_ps.bodies.train.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    val_summaries = batch_proc(open(path + "/data_ps.descriptions.valid.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    val_codes = batch_proc(open(path + "/data_ps.bodies.valid.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    test_summaries = batch_proc(open(path + "/data_ps.descriptions.test.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    test_codes = batch_proc(open(path + "/data_ps.bodies.test.txt", encoding='utf-8', errors='ignore').readlines(), preprocess_language)
    return train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes


def load_iyer_file(filename: str) -> Tuple[List[str], List[str]]:
    dataset = load_iyer_dataset(filename)
    summaries = [example[0] for example in dataset]
    codes = [example[1] for example in dataset]
    return summaries, codes


def load_iyer_dataset(filename: str, alternate_summaries_filename: str = None) -> List[Tuple[str, str]]:
    file_contents = open(filename).readlines()
    if alternate_summaries_filename:
        alternate_file_contents = open(alternate_summaries_filename).readlines()
    dataset = []
    for line_num in range(len(file_contents)):
        line = file_contents[line_num]
        split_line = line.split('\t')
        if len(split_line) == 5:
            summary = preprocess_language(split_line[2])
            code = preprocess_source_code(split_line[3])
            if alternate_summaries_filename is None:
                dataset.append((summary, code))
            else:
                alternate_summaries = []
                for alternate_idx in range(line_num + len(file_contents), len(alternate_file_contents),
                                           len(file_contents)):
                    split_alt_line = alternate_file_contents[alternate_idx].split('\t')
                    if len(split_alt_line) == 2:
                        alternate_summary = preprocess_language(split_alt_line[1])
                        alternate_summaries.append(alternate_summary)
                dataset.append((summary, code, alternate_summaries))
    return dataset


def load_csv_dataset(filename: str) -> List[Tuple[str, str]]:
    file = open(filename, encoding='UTF8')
    reader = csv.reader(file)
    dataset = []
    for row in reader:
        summary = preprocess_language(row[0])
        code = preprocess_source_code(row[1])
        dataset.append((summary, code))
    return dataset


def parse_code(code: str, max_len: int = 1000) -> List[str]:
    lexer = CSharp4Lexer(InputStream(code))
    token_stream = CommonTokenStream(lexer)
    token_stream.fetch(3 * max_len)  # We fetch more than the max len so we can detect oversize codes later...
    parsed_code = ['<s>']
    for token in token_stream.tokens:
        if token.type == 109:
            parsed_code += ["CODE_INTEGER"]
        elif token.type == 111:
            parsed_code += ["CODE_REAL"]
        elif token.type == 112:
            parsed_code += ["CODE_CHAR"]
        elif token.type == 113:
            parsed_code += ["CODE_STRING"]
        elif token.type == -1:
            parsed_code += ["</s>"]
            break
        elif token.type in [4, 5, 6, 7, 8, 9]:  # whitespace and comments and newline
            pass
        else:
            parsed_code += [str(token.text)]
    return parsed_code


def parse_codes(codes: List[str], max_len: int = 1000) -> List[List[str]]:
    parsed_codes = []
    for code in codes:
        parsed_code = parse_code(code, max_len=max_len)
        parsed_codes.append(parsed_code)
    return parsed_codes


def tokenize_texts(texts: List[str]) -> List[List[str]]:
    tokenized = []
    for text in texts:
        tokenized.append(tokenize_text(text))
    return tokenized


def eof_text(text: str) -> str:
    text = "<s>" + text + "</s>"
    return text


def eof_texts(texts: List[str]) -> List[str]:
    texts = [eof_text(text) for text in texts]
    return texts


def process_dataset(summaries, codes, language_seqifier, code_seqifier, l_dim, c_dim,
                    oversize_sequence_behavior='leave_out'):
    assert len(summaries) == len(codes)
    summaries_seq = language_seqifier.tokenize_texts(summaries)
    codes_seq = code_seqifier.tokenize_texts(codes)
    summaries_trim, codes_trim = sequences_to_tensors(summaries_seq, codes_seq, l_dim, c_dim,
                                                      oversize_sequence_behavior=oversize_sequence_behavior)
    return summaries_trim, codes_trim


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
