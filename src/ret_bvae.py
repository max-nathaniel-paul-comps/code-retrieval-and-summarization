import sys
from bvae import *
from text_data_utils import *
from seqifier import Seqifier


class RetBVAE(object):
    def __init__(self, model, code_snippets, language_seqifier, code_seqifier, leave_out_oversize=False):
        self.model = model
        self.raw_codes = code_snippets
        codes_seq = code_seqifier.seqify_texts(code_snippets)
        if leave_out_oversize:
            codes_seq = [seq for seq in codes_seq if len(seq) <= model.c_dim]
        codes_padded = pad_sequences(codes_seq, maxlen=model.c_dim, padding='post', value=0)
        codes_embedded = model.source_code_embedding(codes_padded)
        self.codes = model.source_code_encoder(codes_embedded).mean()
        self.code_snippets = code_snippets
        self.language_seqifier = language_seqifier
        self.code_seqifier = code_seqifier

    def get_similarities(self, query):
        query_prep = preprocess_language(query)
        query_seq = self.language_seqifier.seqify_texts([query_prep])
        query_padded = pad_sequences(query_seq, maxlen=self.model.l_dim, padding='post', value=0)
        query_embedded = self.model.language_embedding(query_padded)
        query_encoded = self.model.language_encoder(query_embedded).mean()
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
    model.compile()
    model.load_weights(model_path + "model_checkpoint.ckpt")

    print("Loading test dataset for evaluation...")
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt")
    test_summaries, test_codes = process_dataset(test_summaries, test_codes, language_seqifier, code_seqifier,
                                                 model.l_dim, model.c_dim)
    test_loss = model.evaluate((test_summaries, test_codes), None, verbose=False)
    print("Test loss: " + str(test_loss))

    print("Preparing interactive retrieval demo...")
    dev_summaries, dev_codes = load_iyer_file("../data/iyer_csharp/dev.txt")
    ret_bvae = RetBVAE(model, dev_codes, language_seqifier, code_seqifier, leave_out_oversize=True)

    ret_bvae.interactive_demo()


if __name__ == "__main__":
    main()
