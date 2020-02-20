import tensorflow as tf
import tensorflow_probability as tfp
import random
import gensim
from text_data_utils import *


class VariationalEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, wv_size, name='variational_encoder'):
        hidden_1_dim = int(input_dim * wv_size - (input_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(VariationalEncoder, self).__init__(
            [
                tf.keras.layers.Flatten(input_shape=(input_dim, wv_size)),
                tf.keras.layers.Dense(hidden_1_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(hidden_2_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(latent_dim * 2)
            ],
            name=name
        )
        self.latent_dim = latent_dim

    def call(self, x, training=None, **kwargs):
        dist = super(VariationalEncoder, self).call(x, training=training, **kwargs)
        mean = dist[:, :self.latent_dim]
        stddev = tf.math.abs(dist[:, self.latent_dim:])
        return tfp.distributions.Normal(mean, stddev)


class Decoder(tf.keras.models.Sequential):
    def __init__(self, latent_dim, reconstructed_dim, wv_size, name='decoder'):
        hidden_1_dim = int(reconstructed_dim * wv_size - (reconstructed_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(Decoder, self).__init__(
            [
                tf.keras.layers.Dense(hidden_2_dim, input_dim=latent_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(hidden_1_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(reconstructed_dim * wv_size),
                tf.keras.layers.Reshape((reconstructed_dim, wv_size))
            ],
            name=name
        )


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, wv_size, name='vae'):
        super(VariationalAutoEncoder, self).__init__(name=name)
        self.encoder = VariationalEncoder(input_dim, latent_dim, wv_size)
        self.decoder = Decoder(latent_dim, input_dim, wv_size)
        self.wv_size = wv_size

    def call(self, inputs, training=None, mask=None):
        latent_dists = self.encoder(inputs)
        sample_latent = latent_dists.sample()
        emp_dist = tfp.distributions.Empirical(sample_latent, event_ndims=1)
        kl_divergence = tf.reduce_mean(
            tfp.distributions.kl_divergence(
                tfp.distributions.Normal(emp_dist.mean(), emp_dist.stddev()),
                tfp.distributions.Normal(0.0, 1.0)
            )
        )
        self.add_loss(kl_divergence)
        decoded = self.decoder(sample_latent)
        mask = tf.reduce_all(tf.logical_not(tf.equal(inputs, 0.0)), axis=-1)
        recon_tensor = tf.losses.cosine_similarity(inputs, decoded) + 1
        recon_masked = tf.where(mask, x=recon_tensor, y=0.0)
        recon = tf.reduce_sum(recon_masked) / tf.reduce_sum(tf.cast(mask, 'float32'))
        self.add_loss(recon)
        return decoded


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

    model = VariationalAutoEncoder(train_codes.shape[1], 256, wv_size)
    model.compile(optimizer='adam')
    model.fit(train_codes, None, batch_size=128, epochs=35, validation_data=(val_codes, None))

    for _ in range(20):
        random.seed()
        random_idx = random.randrange(test_codes.shape[0])
        rand_test = np.array([test_codes[random_idx]])
        print("(Test Set) Input: ", tensor_to_tokenized_texts(rand_test, code_wv)[0])
        rec = model.decoder(model.encoder(rand_test).mean()).numpy()
        print("(Test Set) Recon: ", tensor_to_tokenized_texts(rec, code_wv)[0])
        print()


if __name__ == "__main__":
    main()
