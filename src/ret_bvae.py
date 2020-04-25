import tensorflow as tf
import argparse
from bvae import BimodalVariationalAutoEncoder
import text_data_utils as tdu


class RetBVAE(object):
    def __init__(self, model, code_snippets, query_preprocess_method=tdu.preprocess):
        self.model = model
        self.raw_codes = code_snippets
        self.codes = model.codes_to_latent(self.raw_codes).mean()
        self.query_preprocess_method = query_preprocess_method

    def get_similarities(self, query):
        query_prep = self.query_preprocess_method(query)
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
    parser = argparse.ArgumentParser(description="Interactive demo of BVAE-based code retrieval")
    parser.add_argument("--prog_lang", choices=["csharp", "python", "java"], required=True)
    parser.add_argument("--model_path", help="Path to the BVAE", required=True)
    args = vars(parser.parse_args())
    prog_lang = args["prog_lang"]
    model_path = args["model_path"]

    print("Loading model...")
    model = BimodalVariationalAutoEncoder(model_path)

    print("Preparing interactive retrieval demo...")
    if prog_lang == "csharp":
        _, codes = tdu.load_iyer_file("../data/iyer_csharp/dev.txt")
        query_preprocess_method = tdu.preprocess_stackoverflow_summary
    elif prog_lang == "python":
        _, _, test = tdu.load_edinburgh_dataset("../data/edinburgh_python")
        codes = [ex[1] for ex in test]
        query_preprocess_method = tdu.preprocess_edinburgh_python_or_summary
    elif prog_lang == "java":
        dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
        codes = [ex[1] for ex in dataset]
        query_preprocess_method = tdu.preprocess_javadoc
    else:
        raise Exception()

    ret_bvae = RetBVAE(model, codes, query_preprocess_method=query_preprocess_method)
    ret_bvae.interactive_demo()


if __name__ == "__main__":
    main()
