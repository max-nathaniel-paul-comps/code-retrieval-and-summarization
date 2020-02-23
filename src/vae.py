import tensorflow as tf
import tensorflow_probability as tfp
import random
import gensim
import matplotlib.pyplot as plt
from text_data_utils import *


def recon_loss(true, pred):
    mask = tf.reduce_all(tf.logical_not(tf.equal(true, 0.0)), axis=-1)
    recon_tensor = tf.losses.cosine_similarity(true, pred) + 1
    recon_masked = tf.where(mask, x=recon_tensor, y=0.0)
    recon = tf.reduce_sum(recon_masked) / tf.reduce_sum(tf.cast(mask, 'float32'))
    return recon


class VariationalEncoder(tf.keras.models.Sequential):
    def __init__(self, input_dim, latent_dim, wv_size, name='variational_encoder'):
        hidden_1_dim = int(input_dim * wv_size - (input_dim * wv_size - latent_dim) / 2)
        hidden_2_dim = int(hidden_1_dim - (hidden_1_dim - latent_dim) / 2)
        super(VariationalEncoder, self).__init__(
            [
                tf.keras.layers.Flatten(input_shape=(input_dim, wv_size)),
                tf.keras.layers.Dense(hidden_1_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.075),
                tf.keras.layers.Dense(hidden_2_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.075),
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
                tf.keras.layers.Dropout(0.075),
                tf.keras.layers.Dense(hidden_1_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.075),
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
        recon = recon_loss(inputs, decoded)
        self.add_loss(recon)
        return decoded


def main():
    wv = gensim.models.KeyedVectors.load_word2vec_format("../data/embeddings/w2v_format_codes_vectors.txt")
    wv_size = wv.vector_size

    max_len = 100

    _, train = load_iyer_file("../data/iyer_csharp/train.txt", max_len=max_len)
    _, val = load_iyer_file("../data/iyer_csharp/valid.txt", max_len=max_len)
    _, test = load_iyer_file("../data/iyer_csharp/test.txt", max_len=max_len)

    train = tokenized_texts_to_tensor(train, wv, max_len)
    val = tokenized_texts_to_tensor(val, wv, max_len)
    test = tokenized_texts_to_tensor(test, wv, max_len)

    latent_dim = 192
    model = VariationalAutoEncoder(train.shape[1], latent_dim, wv_size)
    model.compile(optimizer='adam')
    history = model.fit(train, None, batch_size=128, epochs=24, validation_data=(val, None))

    print("\nTest Set Loss: %s\n" % model.evaluate(test, None, batch_size=128, verbose=False))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('vae model loss latent_dim=' + str(latent_dim))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    for _ in range(20):
        random.seed()
        random_idx = random.randrange(test.shape[0])
        rand_test = np.array([test[random_idx]])
        print("(Test Set) Input: ", tensor_to_tokenized_texts(rand_test, wv)[0])
        rec = model.decoder(model.encoder(rand_test).mean()).numpy()
        print("(Test Set) Recon: ", tensor_to_tokenized_texts(rec, wv)[0])
        print()


if __name__ == "__main__":
    main()
