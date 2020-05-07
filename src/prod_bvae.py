import tensorflow as tf
import json
import argparse
import os
from bvae import preprocessors
from tokenizer import Tokenizer
from text_data_utils import de_eof_text


class ProductionBVAE(object):

    def __init__(self, model_path):
        model_path = os.path.abspath(model_path) + "/"
        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.c_prep = preprocessors[model_description['c_type']]
        self.l_prep = preprocessors[model_description['l_type']]

        self.c_tok = Tokenizer(model_description['code_tokenizer_type'],
                               model_path + model_description['code_tokenizer_path'])
        self.l_tok = Tokenizer(model_description['language_tokenizer_type'],
                               model_path + model_description['language_tokenizer_path'])

        self.c_dim = model_description['c_dim']
        self.l_dim = model_description['l_dim']

        self.model = tf.saved_model.load(model_path)

    def summarize(self, codes, beam_width=1):
        prepped = list(map(self.c_prep, codes))
        tokenized = list(map(self.c_tok.tokenize_text, prepped))
        padded = tf.keras.preprocessing.sequence.pad_sequences(tokenized, self.c_dim, dtype='int32', padding='post',
                                                               truncating='post')
        latent = self.model.codes_to_latent_core(padded)
        summaries_tok = self.model.latent_to_summaries_core(latent, beam_width)
        summaries_eofed = list(map(self.l_tok.de_tokenize_text, summaries_tok))
        summaries = list(map(de_eof_text, summaries_eofed))
        return summaries


def demonstrate_retrieval(model, batch_size):
    raise NotImplementedError()


def demonstrate_summarization(model, beam_width):
    while True:
        code = input("Input Code: ")
        if code == "quit" or code == "exit":
            break
        summary = model.summarize([code], beam_width)
        print("Generated Summary: %s" % summary)


def main():
    parser = argparse.ArgumentParser(description="Demonstrate a BVAE that is saved in the SavedModel format")
    parser.add_argument("--model_path", help="Path to the model", required=True)
    parser.add_argument("--mode", help="Whether the model is for retrieval or summarization",
                        choices=["retrieval", "summarization"], required=True)
    parser.add_argument("--batch_size", help="Size of batches to feed into the model. Only relevant for retrieval.",
                        type=int, default=64)
    parser.add_argument("--beam_width",
                        help="Beam width to use for beam search decoding. Only relevant for summarization.",
                        type=int, default=1)
    args = vars(parser.parse_args())

    model_path = args["model_path"]
    mode = args["mode"]

    model = ProductionBVAE(model_path)

    if mode == "retrieval":
        batch_size = args["batch_size"]
        demonstrate_retrieval(model, batch_size)
    elif mode == "summarization":
        beam_width = args["beam_width"]
        demonstrate_summarization(model, beam_width)
    else:
        raise Exception()


if __name__ == "__main__":
    main()
