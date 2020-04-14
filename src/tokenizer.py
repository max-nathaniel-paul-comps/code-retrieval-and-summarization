import os
import text_data_utils as tdu
import tensorflow_datasets as tfds
import json


def build_vocab(training_texts, min_count, oov_token='<unk>'):
    vocab = {}
    for text in training_texts:
        for token in text:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    final_vocab = {oov_token: 0}
    i = 1
    for token, count in vocab.items():
        if count >= min_count:
            final_vocab[token] = i
            i += 1
    return final_vocab


def load_vocab_from_file(filename):
    with open(filename) as json_file:
        vocab = json.load(json_file)
    return vocab


def save_vocab_to_file(vocab, filename):
    with open(filename, mode='w') as json_file:
        json.dump(vocab, json_file)


def encode_split_text(split_text, vocab):
    encoded = []
    for token in split_text:
        if token in vocab:
            encoded.append(vocab[token])
        else:
            encoded.append(0)
    return encoded


def decode_text(encoded_text, vocab):
    decoded = []
    for token_id in encoded_text:
        for k, v in vocab.items():
            if token_id == v:
                decoded.append(k)
                break
    return decoded


class Tokenizer(object):

    def __init__(self, seq_type, path, target_vocab_size=None, vocab_min_count=None, training_texts=None):
        """
        :param seq_type: regex_based, antlr_csharp, or subwords
        :param path: relative path to the tokenizer save file
        :param target_vocab_size: Provide if creating a new subwords tokenizer.
        :param vocab_min_count: Provide if creating a new regex_based or antlr_csharp tokenizer.
        :param training_texts: Provide if creating a new tokenizer of any kind.
        """

        if seq_type == 'regex_based' or seq_type == 'antlr_csharp':
            if seq_type == 'regex_based':
                splitter = tdu.tokenize_text
            else:
                splitter = tdu.parse_code

            def de_splitter(text):
                return " ".join(text)

            if not path.endswith(".json"):
                path += ".json"
            if os.path.isfile(path):
                vocab = load_vocab_from_file(path)
            else:
                print("Could not find the tokenizer vocab save file '%s'. Creating the tokenizer..." % path)
                assert training_texts is not None
                split_texts = [splitter(text) for text in training_texts]
                vocab = build_vocab(split_texts, vocab_min_count)
                save_vocab_to_file(vocab, path)

            self.vocab_size = len(vocab)
            self._tokenizer_fn = lambda texts: [encode_split_text(splitter(text), vocab) for text in texts]
            self._de_tokenizer_fn = lambda seqs: [de_splitter(decode_text(seq, vocab)) for seq in seqs]
            self.start_token = vocab['<s>']
            self.end_token = vocab['</s>']

        elif seq_type == 'subwords':
            if os.path.isfile(path + ".subwords"):
                subword_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(path)
            else:
                print("Could not find the tokenizer save file '%s'. Creating the tokenizer..." % (path + ".subwords"))
                subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                    tdu.eof_texts(training_texts), target_vocab_size, reserved_tokens=['<s>', '</s>']
                )
                subword_encoder.save_to_file(path)

            self.vocab_size = subword_encoder.vocab_size
            self._tokenizer_fn = lambda texts: [
                subword_encoder.encode(tdu.eof_text(text)) for text in texts
            ]
            self._de_tokenizer_fn = lambda seqs: [
                subword_encoder.decode(seq) for seq in seqs
            ]
            self.start_token = subword_encoder.encode("<s>")[0]
            self.end_token = subword_encoder.encode("</s>")[0]

        else:
            raise Exception("Invalid seqifier type %s" % seq_type)

    def tokenize_texts(self, texts):
        tokenized = self._tokenizer_fn(texts)
        return tokenized

    def de_tokenize_texts(self, seqs, hide_eos=True):
        for i in range(len(seqs)):
            if hide_eos and seqs[i][0] == self.start_token:
                seqs[i] = seqs[i][1:]
            if hide_eos and seqs[i][-1] == self.end_token:
                seqs[i] = seqs[i][:-1]
        texts = self._de_tokenizer_fn(seqs)
        return texts
