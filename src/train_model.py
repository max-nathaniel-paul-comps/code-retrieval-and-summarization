import matplotlib.pyplot as plt
from bvae import *


def main():
    train_summaries, train_codes, val_summaries, val_codes, test_summaries, test_codes = \
        load_csv_dataset("../data2/processeed_data2.csv")

    language_wv_size = 128
    source_code_wv_size = 128
    language_wv = gensim.models.Word2Vec(train_summaries, size=language_wv_size).wv
    code_wv = gensim.models.Word2Vec(train_codes, size=source_code_wv_size).wv

    language_dim = 39
    source_code_dim = 50

    train_summaries, train_codes = trim_to_len(train_summaries, train_codes, language_dim, source_code_dim)
    val_summaries, val_codes = trim_to_len(val_summaries, val_codes, language_dim, source_code_dim)
    test_summaries, test_codes = trim_to_len(test_summaries, test_codes, language_dim, source_code_dim)

    train_summaries = tokenized_texts_to_tensor(train_summaries, language_wv, language_dim)
    val_summaries = tokenized_texts_to_tensor(val_summaries, language_wv, language_dim)
    test_summaries = tokenized_texts_to_tensor(test_summaries, language_wv, language_dim)

    train_codes = tokenized_texts_to_tensor(train_codes, code_wv, source_code_dim)
    val_codes = tokenized_texts_to_tensor(val_codes, code_wv, source_code_dim)
    test_codes = tokenized_texts_to_tensor(test_codes, code_wv, source_code_dim)

    latent_dim = 128

    model = BimodalVariationalAutoEncoder(language_dim, language_wv_size, source_code_dim, source_code_wv_size,
                                          latent_dim, input_dropout=0.2)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit((train_summaries, train_codes), None, batch_size=128, epochs=30,
                        validation_data=((val_summaries, val_codes), None),
                        callbacks=[reduce_on_plateau, early_stopping])

    model_description = {
        'language_dim': language_dim,
        'language_wv_size': language_wv_size,
        'source_code_dim': source_code_dim,
        'source_code_wv_size': source_code_wv_size,
        'latent_dim': latent_dim,
    }
    with open("saved_model/model_description.json", 'w') as json_file:
        json.dump(model_description, json_file)

    language_wv.save("saved_model/language_wv.txt")
    code_wv.save("saved_model/code_wv.txt")

    model.save_weights("saved_model/model_weights")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss (mpreg) latent_dim=' + str(latent_dim))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    test_loss = model.evaluate((test_summaries, test_codes), None, batch_size=128, verbose=False)
    print("Test loss: " + str(test_loss))


if __name__ == "__main__":
    main()
