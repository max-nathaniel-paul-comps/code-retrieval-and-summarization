import os
import text_data_utils as tdu
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow_datasets.core.features.text import SubwordTextEncoder


class Seqifier(object):

    def __init__(self, seq_type, path, target_vocab_size=None, training_texts=None):

        if seq_type == 'regex_based' or seq_type == 'antlr_csharp':
            if seq_type == 'regex_based':
                splitter = tdu.tokenize_texts
            else:
                splitter = tdu.parse_codes
            if not path.endswith(".json"):
                path += ".json"
            if os.path.isfile(path):
                with open(path, 'r') as json_file:
                    keras_tokenizer = tokenizer_from_json(json_file.read())
            else:
                print("Could not find the seqifier save file '%s'. Creating the seqifier..." % path)
                assert training_texts is not None
                keras_tokenizer = Tokenizer(num_words=target_vocab_size, oov_token='<unk>', filters='', lower=False)
                split_texts = splitter(training_texts)
                keras_tokenizer.fit_on_texts(split_texts)
                out_json = keras_tokenizer.to_json()
                with open(path, 'w') as json_file:
                    json_file.write(out_json)

            self.vocab_size = keras_tokenizer.num_words
            self.seqifier_fn = lambda texts: keras_tokenizer.texts_to_sequences(splitter(texts))
            self.de_seqifier_fn = lambda seqs: keras_tokenizer.sequences_to_texts(seqs)
            self.start_token = keras_tokenizer.texts_to_sequences(["<s>"])[0][0]
            self.end_token = keras_tokenizer.texts_to_sequences(["</s>"])[0][0]

        elif seq_type == 'subwords':
            if os.path.isfile(path + ".subwords"):
                subword_encoder = SubwordTextEncoder.load_from_file(path)
            else:
                print("Could not find the seqifier save file '%s'. Creating the seqifier..." % (path + ".subwords"))
                subword_encoder = SubwordTextEncoder.build_from_corpus(tdu.eof_texts(training_texts), target_vocab_size,
                                                                       reserved_tokens=['<s>', '</s>'])
                subword_encoder.save_to_file(path)

            self.vocab_size = subword_encoder.vocab_size
            self.seqifier_fn = lambda texts: [
                subword_encoder.encode(tdu.eof_text(text)) for text in texts
            ]
            self.de_seqifier_fn = lambda seqs: [
                subword_encoder.decode(seq) for seq in seqs
            ]
            self.start_token = subword_encoder.encode("<s>")[0]
            self.end_token = subword_encoder.encode("</s>")[0]

        else:
            raise Exception("Invalid seqifier type %s" % seq_type)

    def seqify_texts(self, texts):
        seqified = self.seqifier_fn(texts)
        return seqified

    def de_seqify_texts(self, seqs):
        for i in range(len(seqs)):
            if seqs[i][0] == self.start_token:
                seqs[i] = seqs[i][1:]
            if seqs[i][-1] == self.end_token:
                seqs[i] = seqs[i][:-1]
        texts = self.de_seqifier_fn(seqs)
        return texts
