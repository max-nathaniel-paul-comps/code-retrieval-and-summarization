import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from bvae import *


def main():
    train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes = \
        load_csv_dataset("../data2/processeed_data2.csv")

    language_dim = 39
    source_code_dim = 50

    language_tokenizer_file = "language_tokenizer"
    if os.path.isfile(language_tokenizer_file + ".subwords"):
        language_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(language_tokenizer_file)
    else:
        language_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (summary for summary in train_summaries),
            1024, reserved_tokens=['<s>', '</s>'])
        language_tokenizer.save_to_file(language_tokenizer_file)

    code_tokenizer_file = "code_tokenizer"
    if os.path.isfile(code_tokenizer_file + ".subwords"):
        code_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(code_tokenizer_file)
    else:
        code_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (code for code in train_codes),
            1024, reserved_tokens=['<s>', '</s>'])
        code_tokenizer.save_to_file(code_tokenizer_file)

    train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes = subword_encode(
        language_tokenizer, code_tokenizer, language_dim, source_code_dim,
        train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes
    )

    latent_dim = 128

    model = BimodalVariationalAutoEncoder(language_dim, language_tokenizer.vocab_size,
                                          source_code_dim, code_tokenizer.vocab_size,
                                          latent_dim, input_dropout=0.2)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit((train_summaries, train_codes), None, batch_size=64, epochs=30,
                        validation_data=((val_summaries, val_codes), None),
                        callbacks=[reduce_on_plateau, early_stopping])

    model_description = {
        'language_dim': language_dim,
        'source_code_dim': source_code_dim,
        'latent_dim': latent_dim,
    }
    with open("saved_model/model_description.json", 'w') as json_file:
        json.dump(model_description, json_file)

    model.save_weights("saved_model/model_weights")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss (mpreg) latent_dim=' + str(latent_dim))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    test_loss = model.evaluate((test_summaries, test_codes), None, batch_size=64, verbose=False)
    print("Test loss: " + str(test_loss))


if __name__ == "__main__":
    main()
