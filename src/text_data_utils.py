import re
import csv
import json
from typing import Tuple, List


def rephrase_question(x: str) -> str:
    if len(x) < 2:
        return x
    if x[-1] == '?':
        x = x[:-1]
    for opener in ['how do i ', 'how do you ', 'how can i ', 'how to ', 'best way to ', 'can i ',
                   'is there a way to ', 'easiest way to ', 'best implementation for ',
                   'best implementation of ', 'what is the best way to ', 'what is the proper way to ',
                   'is it possible to ', 'would it be possible to '
                   'how ', 'c# how to ', 'c# how ', 'c# - ', 'c# ']:
        if x.lower().startswith(opener):
            x = x[len(opener):]
    for closer in [' in c#', ' with c#', ' using c#', ' c#']:
        if x.lower().endswith(closer):
            x = x[:-len(closer)]
    return x


def preprocess(x: str, remove_stars=False, remove_java_doc_vars=False, remove_html_tags=False, remove_comments=False,
               remove_start_and_end_quotes=False, rephrase=False, lower=False) -> str:
    if remove_java_doc_vars:
        x = re.sub(r'(?<![{])(@.*)', ' ', x)
    if remove_comments:
        x = re.sub(r'(?<![:\"])(//.*?(?:\n|\\n))', ' ', x)
    if remove_html_tags:
        x = re.sub(r'<.*?>', ' ', x)
    x = x.replace('\\n', ' ').replace('\n', ' ')
    x = x.replace('\\t', ' ').replace('\t', ' ')
    if remove_stars:
        x = x.replace('/*', ' ').replace('*/', ' ').replace('*', ' ')
    if remove_start_and_end_quotes:
        if x.startswith('\''):
            x = x[len('\''):]
        if x.endswith('\''):
            x = x[:-len('\'')]
        if x.startswith('"'):
            x = x[len('"'):]
        if x.endswith('"'):
            x = x[:-len('"')]
    x = x.strip()
    x = re.sub(r'(\s\s+)', ' ', x)
    if rephrase:
        x = rephrase_question(x)
    if lower:
        x = x.lower()
    return x


def preprocess_csharp_or_java(x: str) -> str:
    return preprocess(x, remove_comments=True, remove_start_and_end_quotes=True)


def preprocess_javadoc(x: str) -> str:
    return preprocess(x, remove_stars=True, remove_java_doc_vars=True, remove_html_tags=True)


def preprocess_stackoverflow_summary(x: str) -> str:
    return preprocess(x, rephrase=True, remove_html_tags=True, lower=True, remove_start_and_end_quotes=True)


def preprocess_edinburgh_python_or_summary(x: str) -> str:
    return preprocess(x, remove_start_and_end_quotes=True)


def tokenize_text(text: str) -> List[str]:
    """
    Splits a text into tokens using a simple regex, and adds start and end tokens.
    """
    words_re = re.compile(r'(\w+|[^\w\s])')
    return ['<s>'] + words_re.findall(text) + ['</s>']


def tokenize_texts(texts: List[str]) -> List[List[str]]:
    return [tokenize_text(text) for text in texts]


def edinburgh_dataset_as_generator(summaries_path: str, codes_path: str):
    summaries_file = open(summaries_path, encoding='utf-8', errors='ignore')
    codes_file = open(codes_path, encoding='utf-8', errors='ignore')

    def generator():
        while True:
            summary = summaries_file.readline()
            code = codes_file.readline()
            if len(summary) == 0:
                assert len(code) == 0
                summaries_file.seek(0)
                codes_file.seek(0)
                break
            assert len(code) > 0
            summary_prepped = preprocess_edinburgh_python_or_summary(summary)
            code_prepped = preprocess_edinburgh_python_or_summary(code)
            yield summary_prepped, code_prepped

    return generator


def load_edinburgh_dataset(path: str):
    train = list(edinburgh_dataset_as_generator(path + "/data_ps.descriptions.train.txt",
                                                path + "/data_ps.declbodies.train.txt")())
    val = list(edinburgh_dataset_as_generator(path + "/data_ps.descriptions.valid.txt",
                                              path + "/data_ps.declbodies.valid.txt")())
    test = list(edinburgh_dataset_as_generator(path + "/data_ps.descriptions.test.txt",
                                               path + "/data_ps.declbodies.test.txt")())
    return train, val, test


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
            summary = preprocess_stackoverflow_summary(split_line[2])
            code = preprocess_csharp_or_java(split_line[3])
            if alternate_summaries_filename is None:
                dataset.append((summary, code))
            else:
                alternate_summaries = []
                for alternate_idx in range(line_num + len(file_contents), len(alternate_file_contents),
                                           len(file_contents)):
                    split_alt_line = alternate_file_contents[alternate_idx].split('\t')
                    if len(split_alt_line) == 2:
                        alternate_summary = preprocess_stackoverflow_summary(split_alt_line[1])
                        alternate_summaries.append(alternate_summary)
                dataset.append((summary, code, alternate_summaries))
    return dataset


def load_csv_dataset(filename: str) -> List[Tuple[str, str]]:
    file = open(filename, encoding='UTF8')
    reader = csv.reader(file)
    dataset = []
    for row in reader:
        summary = preprocess_stackoverflow_summary(row[0])
        code = preprocess_csharp_or_java(row[1])
        dataset.append((summary, code))
    return dataset


def json_java_dataset_as_generator(filename):
    file = open(filename, mode='r', encoding='utf-8')

    def generator():
        while True:
            row = file.readline()
            if len(row) == 0:
                file.seek(0)
                break
            json_row = json.loads(row)
            summary = preprocess_javadoc(json_row["nl"])
            code = preprocess_csharp_or_java(json_row["code"])
            yield summary, code

    return generator


def load_json_dataset(filename):
    generator = json_java_dataset_as_generator(filename)
    return list(generator())


def generator_from_list(rows):
    def generator():
        for row in rows:
            yield row
    return generator


def eof_text(text: str) -> str:
    text = "<s>" + text + "</s>"
    return text


def de_eof_text(text: str) -> str:
    if text.startswith("<s>"):
        text = text[len("<s>"):]
    if text.endswith("</s>"):
        text = text[:-len("</s>")]
    return text


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
