import gensim
import random
import matplotlib.pyplot as plt
from vae import *
from text_data_utils import *


class BimodalVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, language_dim, source_code_dim, latent_dim, wv_size, name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=name)
        self.language_encoder = VariationalEncoder(language_dim, latent_dim, wv_size, name='language_encoder')
        self.source_code_encoder = VariationalEncoder(source_code_dim, latent_dim, wv_size, name='source_code_encoder')
        self.language_decoder = Decoder(latent_dim, language_dim, wv_size, name='language_decoder')
        self.source_code_decoder = Decoder(latent_dim, source_code_dim, wv_size, name='source_code_decoder')

    def call(self, inputs, training=None, mask=None):
        language_batch = inputs[0]
        source_code_batch = inputs[1]
        enc_language_dists = self.language_encoder(language_batch)
        enc_source_code_dists = self.source_code_encoder(source_code_batch)
        mean_mean = (enc_language_dists.mean() + enc_source_code_dists.mean()) / 2
        mean_stddev = (enc_language_dists.stddev() + enc_source_code_dists.stddev()) / 2

        language_kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                enc_language_dists,
                tfp.distributions.Normal(mean_mean, mean_stddev)
            )
        )
        self.add_loss(language_kl_divergence)

        source_code_kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                enc_source_code_dists,
                tfp.distributions.Normal(mean_mean, mean_stddev)
            )
        )
        self.add_loss(source_code_kl_divergence)

        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()
        dec_language = self.language_decoder(enc_language)
        dec_source_code = self.source_code_decoder(enc_source_code)

        language_mask = tf.reduce_all(tf.logical_not(tf.equal(language_batch, 0.0)), axis=-1)
        language_recon_tensor = tf.losses.cosine_similarity(language_batch, dec_language) + 1
        language_recon_masked = tf.where(language_mask, x=language_recon_tensor, y=0.0)
        language_recon = tf.reduce_sum(language_recon_masked) / tf.reduce_sum(tf.cast(language_mask, 'float32'))
        self.add_loss(language_recon)

        source_code_mask = tf.reduce_all(tf.logical_not(tf.equal(source_code_batch, 0.0)), axis=-1)
        source_code_recon_tensor = tf.losses.cosine_similarity(source_code_batch, dec_source_code) + 1
        source_code_recon_masked = tf.where(source_code_mask, x=source_code_recon_tensor, y=0.0)
        source_code_recon = tf.reduce_sum(source_code_recon_masked) / tf.reduce_sum(tf.cast(source_code_mask, 'float32'))
        self.add_loss(source_code_recon)

        return dec_language, dec_source_code


def main():
    language_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_summaries_vectors.txt")
    code_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_codes_vectors.txt")
    assert language_wv.vector_size == code_wv.vector_size
    wv_size = language_wv.vector_size

    max_len = 50
    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt", max_len=max_len)
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt", max_len=max_len)
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt", max_len=max_len)

    train_summaries = tokenized_texts_to_tensor(train_summaries, language_wv, max_len)
    val_summaries = tokenized_texts_to_tensor(val_summaries, language_wv, max_len)
    test_summaries = tokenized_texts_to_tensor(test_summaries, language_wv, max_len)

    train_codes = tokenized_texts_to_tensor(train_codes, code_wv, max_len)
    val_codes = tokenized_texts_to_tensor(val_codes, code_wv, max_len)
    test_codes = tokenized_texts_to_tensor(test_codes, code_wv, max_len)

    latent_dim = 256
    model = BimodalVariationalAutoEncoder(train_summaries.shape[1], train_codes.shape[1], latent_dim, wv_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=6,
                        validation_data=((val_summaries, val_codes), None))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    for _ in range(20):
        random.seed()
        random_idx = random.randrange(test_summaries.shape[0])
        rand_test_summary = np.array([test_summaries[random_idx]])
        print("(Test Set) Input Summary: ", tensor_to_tokenized_texts(rand_test_summary, language_wv)[0])
        rand_test_code = np.array([test_codes[random_idx]])
        print("(Test Set) Input Code: ", tensor_to_tokenized_texts(rand_test_code, code_wv)[0])
        rec_summary = model.language_decoder(model.language_encoder(rand_test_summary).mean()).numpy()
        print("(Test Set) Reconstructed Summary: ", tensor_to_tokenized_texts(rec_summary, language_wv)[0])
        rec_code = model.source_code_decoder(model.source_code_encoder(rand_test_code).mean()).numpy()
        print("(Test Set) Reconstructed Source Code: ", tensor_to_tokenized_texts(rec_code, code_wv)[0])
        rec_summary_hard = model.language_decoder(model.source_code_encoder(rand_test_code).mean()).numpy()
        print("(Test Set) Reconstructed Summary From Source Code: ", tensor_to_tokenized_texts(rec_summary_hard, language_wv)[0])
        rec_code_hard = model.source_code_decoder(model.language_encoder(rand_test_summary).mean()).numpy()
        print("(Test Set) Reconstructed Source Code From Summary: ", tensor_to_tokenized_texts(rec_code_hard, code_wv)[0])
        print()


if __name__ == "__main__":
    main()
