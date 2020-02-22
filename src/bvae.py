import gensim
import random
import matplotlib.pyplot as plt
from vae import *
from text_data_utils import *


def preg_loss(dists_a, dists_b):
    kl_divergence = tf.reduce_mean(
        tfp.distributions.kl_divergence(
            dists_a,
            dists_b
        )
    )
    logged_kld = tf.math.log(kl_divergence + 1)
    return logged_kld


def dists_means(dists_a, dists_b):
    mean_mean = (dists_a.mean() + dists_b.mean()) / 2
    mean_stddev = (dists_a.stddev() + dists_b.stddev())
    mean_dists = tfp.distributions.Normal(mean_mean, mean_stddev)
    return mean_dists


def mpreg_loss(dists, mean_dist):
    kl_divergence = tf.reduce_mean(
        tfp.distributions.kl_divergence(
            dists,
            mean_dist
        )
    )
    logged_kld = tf.math.log(kl_divergence + 1)
    return logged_kld


class BimodalVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, language_dim, source_code_dim, latent_dim, wv_size, name='bvae'):
        super(BimodalVariationalAutoEncoder, self).__init__(name=name)
        self.language_encoder = VariationalEncoder(language_dim, latent_dim, wv_size, name='language_encoder')
        self.source_code_encoder = VariationalEncoder(source_code_dim, latent_dim, wv_size, name='source_code_encoder')
        self.language_decoder = Decoder(latent_dim, language_dim, wv_size, name='language_decoder')
        self.source_code_decoder = Decoder(latent_dim, source_code_dim, wv_size, name='source_code_decoder')

    def compute_and_add_loss(self, language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                             dec_language, dec_source_code):
        mean_dists = dists_means(enc_language_dists, enc_source_code_dists)
        language_kld = mpreg_loss(enc_language_dists, mean_dists)
        source_code_kld = mpreg_loss(enc_source_code_dists, mean_dists)
        """language_kld = preg_loss(enc_language_dists, enc_source_code_dists)
        source_code_kld = preg_loss(enc_source_code_dists, enc_language_dists)"""
        language_recon = recon_loss(language_batch, dec_language)
        source_code_recon = recon_loss(source_code_batch, dec_source_code)
        self.add_loss(language_kld + source_code_kld + language_recon + source_code_recon)

    def call(self, inputs, training=None, mask=None):
        language_batch = inputs[0]
        source_code_batch = inputs[1]
        enc_language_dists = self.language_encoder(language_batch)
        enc_source_code_dists = self.source_code_encoder(source_code_batch)
        enc_language = enc_language_dists.sample()
        enc_source_code = enc_source_code_dists.sample()
        dec_language = self.language_decoder(enc_language)
        dec_source_code = self.source_code_decoder(enc_source_code)
        self.compute_and_add_loss(language_batch, source_code_batch, enc_source_code_dists, enc_language_dists,
                                  dec_language, dec_source_code)
        return dec_language, dec_source_code


def main():
    language_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_summaries_vectors.txt")
    code_wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_codes_vectors.txt")
    assert language_wv.vector_size == code_wv.vector_size
    wv_size = language_wv.vector_size

    max_len = 100
    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt", max_len=max_len)
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt", max_len=max_len)
    test_summaries, test_codes = load_iyer_file("../data/iyer_csharp/test.txt", max_len=max_len)

    train_summaries = tokenized_texts_to_tensor(train_summaries, language_wv, max_len)
    val_summaries = tokenized_texts_to_tensor(val_summaries, language_wv, max_len)
    test_summaries = tokenized_texts_to_tensor(test_summaries, language_wv, max_len)

    train_codes = tokenized_texts_to_tensor(train_codes, code_wv, max_len)
    val_codes = tokenized_texts_to_tensor(val_codes, code_wv, max_len)
    test_codes = tokenized_texts_to_tensor(test_codes, code_wv, max_len)

    latent_dim = 128
    model = BimodalVariationalAutoEncoder(train_summaries.shape[1], train_codes.shape[1], latent_dim, wv_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=6,
                        validation_data=((val_summaries, val_codes), None))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss (mpreg)')
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
