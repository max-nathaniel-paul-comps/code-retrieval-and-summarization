import sys
from bvae import *
from text_data_utils import *


class RetBVAE(object):
    def __init__(self, model, code_snippets):
        self.model = model
        self.raw_codes = code_snippets
        self.codes = model.codes_to_latent(self.raw_codes).mean()

    def get_similarities(self, query):
        query_prep = preprocess_language(query)
        query_encoded = self.model.summaries_to_latent([query_prep]).mean()
        similarities = tf.losses.cosine_similarity(query_encoded, self.codes, axis=-1) + 1
        return similarities

    def rank_options(self, query):
        similarities = self.get_similarities(query)
        ranked_indices = tf.argsort(similarities, direction='ASCENDING').numpy()
        return ranked_indices

    def interactive_demo(self):
        while True:
            print()
            input_summary = input("Input Summary: ")
            if input_summary == "exit":
                break
            ranked_options = self.rank_options(input_summary)
            print("Retrieved Code: %s" % self.raw_codes[ranked_options[0]])


def main():
    assert len(sys.argv) == 3, "Usage: python ret_bvae.py prog_lang path/to/model/dir/"
    prog_lang = sys.argv[1]
    model_path = sys.argv[2]

    print("Loading model...")
    model = BimodalVariationalAutoEncoder(model_path)

    print("Preparing interactive retrieval demo...")
    if prog_lang == "csharp":
        _, codes = load_iyer_file("../data/iyer_csharp/dev.txt")
    elif prog_lang == "python":
        _, _, test = load_edinburgh_dataset("../data/edinburgh_python")
        codes = [ex[1] for ex in test]
    else:
        raise Exception("Invalid programming language: %s" % prog_lang)

    ret_bvae = RetBVAE(model, codes)
    ret_bvae.interactive_demo()


if __name__ == "__main__":
    main()
