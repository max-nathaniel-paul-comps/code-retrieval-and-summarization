from bvae import *


class RetBVAE(object):
    def __init__(self, code_snippets, model_path="../models/r2/", tokenizers_path="../data/iyer_csharp/"):

        self._language_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
            tokenizers_path + "language_tokenizer")
        code_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizers_path + "code_tokenizer")

        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.l_dim = model_description['l_dim']
        l_emb_dim = model_description['l_emb_dim']
        c_dim = model_description['c_dim']
        c_emb_dim = model_description['c_emb_dim']
        latent_dim = model_description['latent_dim']
        dropout_rate = model_description['dropout_rate']
        architecture = model_description['architecture']

        self._model = BimodalVariationalAutoEncoder(self.l_dim, self._language_tokenizer.vocab_size, l_emb_dim,
                                                    c_dim, code_tokenizer.vocab_size, c_emb_dim,
                                                    latent_dim, input_dropout=dropout_rate, architecture=architecture)

        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

        self._model.load_weights(model_path + "model_checkpoint.ckpt")

        self._code_snippets = [preprocess_source_code(code) for code in code_snippets]
        snippets_tokenized = [code_tokenizer.encode(code) for code in self._code_snippets]
        snippets_padded = tf.keras.preprocessing.sequence.pad_sequences(snippets_tokenized, maxlen=c_dim,
                                                                        padding='post', value=0)
        self._encoded_code_snippets = self._model.source_code_encoder(snippets_padded).mean()

    def _compute_similarities(self, input_summaries):
        summaries = [preprocess_language(summary) for summary in input_summaries]
        summaries_tokenized = [self._language_tokenizer.encode(summary) for summary in summaries]
        summaries_padded = tf.keras.preprocessing.sequence.pad_sequences(summaries_tokenized, maxlen=self.l_dim,
                                                                         padding='post', value=0)
        encoded_summaries = self._model.language_encoder(summaries_padded).mean()
        similarities = tf.matmul(
            tf.math.l2_normalize(encoded_summaries, axis=-1),
            tf.math.l2_normalize(self._encoded_code_snippets, axis=-1),
            transpose_b=True
        )
        return similarities

    def retrieve(self, input_summaries):
        similarities = self._compute_similarities(input_summaries)
        best_snippet_ids = tf.argmax(similarities, axis=-1).numpy()
        best_snippets = [self._code_snippets[best_snippet_ids[i]] for i in range(len(best_snippet_ids))]
        return best_snippets

    def evaluate_retrieval(self, input_summaries, correct_codes):
        similarities = self._compute_similarities(input_summaries)
        sorted_indices = tf.argsort(similarities, axis=-1, direction='DESCENDING')

        reciprocal_ranks = []
        for i in range(len(similarities)):
            for j in range(len(similarities[i])):
                if self._code_snippets[sorted_indices[i][j]] == correct_codes[i]:
                    reciprocal_ranks.append(1.0 / (j + 1.0))
                    break
        mean_reciprocal_rank = np.mean(reciprocal_ranks)

        return mean_reciprocal_rank


def main():
    summaries, codes = load_iyer_file("../data/iyer_csharp/test.txt")
    sample_summaries = summaries[:50]
    sample_correct_codes = codes[:50]
    retriever = RetBVAE(sample_correct_codes)
    retrieved_codes = retriever.retrieve(sample_summaries)

    for i in range(len(sample_summaries)):
        print("Input Summary: " + sample_summaries[i])
        print("Retrieved Code: " + retrieved_codes[i])
        print("Correct Code: " + sample_correct_codes[i])
        print()

    mrr = retriever.evaluate_retrieval(sample_summaries, sample_correct_codes)
    print("MRR: %s" % mrr)


if __name__ == "__main__":
    main()
