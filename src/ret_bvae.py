import sys
from bvae import *
from text_data_utils import *
from seqifier import Seqifier


class RetBVAE(object):
    def __init__(self, model, code_snippets, language_seqifier, code_seqifier):
        self.model = model
        self.raw_codes = code_snippets
        codes_seq = code_seqifier.seqify_texts(code_snippets)
        self.codes = model.source_code_encoder(codes_seq).mean()
        self.code_snippets = code_snippets
        self.language_seqifier = language_seqifier
        self.code_seqifier = code_seqifier

    def get_similarities(self, query):
        query_prep = preprocess_language(query)
        query_seq = self.language_seqifier.seqify_texts([query_prep])
        query_encoded = self.model.language_encoder(query_seq).mean()
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

    print("Loading seqifiers, which are responsible for turning texts into sequences of integers...")
    with open(model_path + "seqifiers_description.json") as seq_desc_json:
        seqifiers_description = json.load(seq_desc_json)
    language_seqifier = Seqifier(seqifiers_description['language_seq_type'],
                                 model_path + seqifiers_description['language_seq_path'])
    code_seqifier = Seqifier(seqifiers_description['source_code_seq_type'],
                             model_path + seqifiers_description['source_code_seq_path'])

    print("Loading model...")
    model = BimodalVariationalAutoEncoder(model_path, language_seqifier, code_seqifier)

    print("Loading test dataset for evaluation...")
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt")
    test_summaries = language_seqifier.seqify_texts(test_summaries)
    test_codes = code_seqifier.seqify_texts(test_codes)

    test_loss = model.evaluate(test_summaries, test_codes)
    print("Test loss: %s" % test_loss.numpy())

    print("Preparing interactive retrieval demo...")
    dev_summaries, dev_codes = load_iyer_file("../data/iyer_csharp/dev.txt")
    ret_bvae = RetBVAE(model, dev_codes, language_seqifier, code_seqifier)

    ret_bvae.interactive_demo()


if __name__ == "__main__":
    main()
