from bvae import *


class RetBVAE(object):
    def __init__(self, model_path="../models/r2/", tokenizers_path="../data/iyer_csharp/"):

        self._language_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
            tokenizers_path + "language_tokenizer")
        self._code_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
            tokenizers_path + "code_tokenizer")

        if not os.path.isfile(model_path + "model_description.json"):
            raise FileNotFoundError("Model description not found")

        with open(model_path + "model_description.json", 'r') as json_file:
            model_description = json.load(json_file)

        self.l_dim = model_description['l_dim']
        l_emb_dim = model_description['l_emb_dim']
        self.c_dim = model_description['c_dim']
        c_emb_dim = model_description['c_emb_dim']
        latent_dim = model_description['latent_dim']
        dropout_rate = model_description['dropout_rate']
        architecture = model_description['architecture']

        self._model = BimodalVariationalAutoEncoder(self.l_dim, self._language_tokenizer.vocab_size, l_emb_dim,
                                                    self.c_dim, self._code_tokenizer.vocab_size, c_emb_dim,
                                                    latent_dim, input_dropout=dropout_rate, architecture=architecture)

        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

        self._model.load_weights(model_path + "model_checkpoint.ckpt")

    def retrieve(self, input_summary, possible_code_snippets):
        summary = preprocess_language(input_summary)
        summary_tokenized = self._language_tokenizer.encode(summary)
        summary_padded = tf.keras.preprocessing.sequence.pad_sequences([summary_tokenized], maxlen=self.l_dim,
                                                                       padding='post', value=0)
        encoded_summary = self._model.language_encoder(summary_padded).mean()

        codes = [preprocess_source_code(code) for code in possible_code_snippets]
        codes_tokenized = [self._code_tokenizer.encode(code) for code in codes]
        codes_padded = tf.keras.preprocessing.sequence.pad_sequences(codes_tokenized, maxlen=self.c_dim,
                                                                     padding='post', value=0)
        encoded_codes = self._model.source_code_encoder(codes_padded).mean()

        similarities = tf.matmul(
            tf.math.l2_normalize(encoded_codes, axis=-1),
            tf.math.l2_normalize(encoded_summary, axis=-1),
            transpose_b=True
        )

        return tf.squeeze(similarities)

    def rank_options(self, input_summary, possible_code_snippets):
        similarities = self.retrieve(input_summary, possible_code_snippets)
        ranked_indices = tf.argsort(similarities, direction='DESCENDING').numpy()
        return ranked_indices

    def interactive_demo(self, code_snippets):
        codes = [preprocess_source_code(code) for code in code_snippets]
        codes_tokenized = [self._code_tokenizer.encode(code) for code in codes]
        codes_padded = tf.keras.preprocessing.sequence.pad_sequences(codes_tokenized, maxlen=self.c_dim,
                                                                     padding='post', value=0)
        encoded_codes = self._model.source_code_encoder(codes_padded).mean()

        while True:
            input_summary = input("Input Summary: ")
            if input_summary == "exit":
                break
            summary = preprocess_language(input_summary)
            summary_tokenized = self._language_tokenizer.encode(summary)
            summary_padded = tf.keras.preprocessing.sequence.pad_sequences([summary_tokenized], maxlen=self.l_dim,
                                                                           padding='post', value=0)
            encoded_summary = self._model.language_encoder(summary_padded).mean()

            similarities = tf.matmul(
                tf.math.l2_normalize(encoded_codes, axis=-1),
                tf.math.l2_normalize(encoded_summary, axis=-1),
                transpose_b=True
            )

            most_similar = tf.argmax(tf.squeeze(similarities))
            print("Retrieved Code: %s" % codes[most_similar])


def main():
    _, codes = load_iyer_file("../data/iyer_csharp/test.txt")
    ret_bvae = RetBVAE()
    ret_bvae.interactive_demo(codes)


if __name__ == "__main__":
    main()
