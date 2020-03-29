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
    assert len(sys.argv) == 2, "Usage: python ret_bvae.py path/to/model/dir/"
    model_path = sys.argv[1]

    print("Loading model...")
    model = BimodalVariationalAutoEncoder(model_path)

    print("Preparing interactive retrieval demo...")
    dev_summaries, dev_codes = load_iyer_file("../data/iyer_csharp/dev.txt")
    ret_bvae = RetBVAE(model, dev_codes)

    ret_bvae.interactive_demo()


if __name__ == "__main__":
    main()
