import matplotlib.pyplot as plt
import random
from bvae import *


def train_bvae(model_save_path="../models/saved_model/", dataset_path="../data/iyer_csharp/",
               l_dim=40, l_vocab_size=5000, l_emb_dim=128, c_dim=60, c_vocab_size=5000, c_emb_dim=128,
               latent_dim=128, dropout_rate=0.1):

    language_tokenizer_file = dataset_path + "language_tokenizer"
    if os.path.isfile(language_tokenizer_file + ".subwords"):
        language_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(language_tokenizer_file)
    else:
        language_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (summary for summary in load_iyer_file(dataset_path + "train.txt")[0]), l_vocab_size, reserved_tokens=['<s>', '</s>'])
        language_tokenizer.save_to_file(language_tokenizer_file)

    code_tokenizer_file = dataset_path + "code_tokenizer"
    if os.path.isfile(code_tokenizer_file + ".subwords"):
        code_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(code_tokenizer_file)
    else:
        code_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (code for code in load_iyer_file(dataset_path + "train.txt")[1]), c_vocab_size, reserved_tokens=['<s>', '</s>'])
        code_tokenizer.save_to_file(code_tokenizer_file)

    train_summaries, train_codes = load_iyer_file(dataset_path + "train.txt")
    val_summaries, val_codes = load_iyer_file(dataset_path + "valid.txt")
    test_summaries, test_codes = load_iyer_file(dataset_path + "test.txt")

    train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes = subword_encode(
        language_tokenizer, code_tokenizer, l_dim, c_dim,
        train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes
    )

    model = BimodalVariationalAutoEncoder(l_dim, language_tokenizer.vocab_size, l_emb_dim,
                                          c_dim, code_tokenizer.vocab_size, c_emb_dim,
                                          latent_dim, input_dropout=dropout_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=25,
                        validation_data=((val_summaries, val_codes), None),
                        callbacks=[reduce_on_plateau, early_stopping])

    model_description = {
        'language_dim': l_dim,
        'l_emb_dim': l_emb_dim,
        'source_code_dim': c_dim,
        'c_emb_dim': c_emb_dim,
        'latent_dim': latent_dim,
    }
    with open(model_save_path + "model_description.json", 'w') as json_file:
        json.dump(model_description, json_file)

    model.save_weights(model_save_path + "model_weights")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss rec_l_dec latent_dim=' + str(latent_dim) + ' d=' + str(dropout_rate))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    test_loss = model.evaluate((test_summaries, test_codes), None, batch_size=128, verbose=False)
    print("Test loss: " + str(test_loss))

    for _ in range(20):
        print()
        random.seed()
        random_idx = random.randrange(test_summaries.shape[0])
        rand_test_summary = test_summaries[random_idx]
        print("(Test Set) Input Summary: ", language_tokenizer.decode(rand_test_summary))
        rand_test_code = test_codes[random_idx]
        print("(Test Set) Input Code: ", code_tokenizer.decode(rand_test_code))
        rec_summary = model.language_decoder(model.language_encoder(np.array([rand_test_summary])).mean())[0].numpy()
        print("(Test Set) Reconstructed Summary: ", language_tokenizer.decode(np.argmax(rec_summary, axis=-1)))
        rec_code = model.source_code_decoder(model.source_code_encoder(np.array([rand_test_code])).mean())[0].numpy()
        print("(Test Set) Reconstructed Source Code: ", code_tokenizer.decode(np.argmax(rec_code, axis=-1)))
        rec_summary_hard = model.language_decoder(model.source_code_encoder(np.array([rand_test_code])).mean())[0].numpy()
        print("(Test Set) Reconstructed Summary From Source Code: ", language_tokenizer.decode(np.argmax(rec_summary_hard, axis=-1)))
        rec_code_hard = model.source_code_decoder(model.language_encoder(np.array([rand_test_summary])).mean())[0].numpy()
        print("(Test Set) Reconstructed Source Code From Summary: ", code_tokenizer.decode(np.argmax(rec_code_hard, axis=-1)))


if __name__ == "__main__":
    train_bvae()
