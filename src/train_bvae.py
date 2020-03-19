import sys
from bvae import *
from text_data_utils import *
from seqifier import *


def train_bvae(model, model_path, train_summaries, train_codes, val_summaries, val_codes):

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.abspath(model_path + "tboard"), histogram_freq=1)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(model_path + 'model_checkpoint.ckpt',
                                                     verbose=True, save_best_only=True,
                                                     monitor='val_loss', save_freq='epoch')
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.fit((train_summaries, train_codes), None, batch_size=128, epochs=100,
              validation_data=((val_summaries, val_codes), None),
              callbacks=[tboard_callback, checkpoints, reduce_on_plateau, early_stopping])


def main():
    assert len(sys.argv) == 2, "Usage: python train_bvae.py path/to/model/dir/"
    model_path = sys.argv[1]

    print("Loading dataset...")
    train_summaries, train_codes = load_iyer_file("../data/iyer_csharp/train.txt")
    val_summaries, val_codes = load_iyer_file("../data/iyer_csharp/valid.txt")

    print("Loading seqifiers, which are responsible for turning texts into sequences of integers...")
    with open(model_path + "seqifiers_description.json") as seq_desc_json:
        seqifiers_description = json.load(seq_desc_json)
    language_seqifier = Seqifier(seqifiers_description['language_seq_type'],
                                 model_path + seqifiers_description['language_seq_path'],
                                 training_texts=train_summaries,
                                 target_vocab_size=seqifiers_description['language_target_vocab_size'])
    code_seqifier = Seqifier(seqifiers_description['source_code_seq_type'],
                             model_path + seqifiers_description['source_code_seq_path'],
                             training_texts=train_codes,
                             target_vocab_size=seqifiers_description['source_code_target_vocab_size'])

    print("Creating model from JSON description...")
    model = BimodalVariationalAutoEncoder(model_path, language_seqifier, code_seqifier)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    if os.path.isfile(model_path + "model_checkpoint.ckpt"):
        model.load_weights(model_path + "model_checkpoint.ckpt")

    if os.path.isfile(model_path + "checkpoint"):
        print("The model has already been trained, and training will continue.")
    else:
        print("The model has not been trained yet.")

    print("Preparing datasets for training...")
    train_summaries, train_codes = process_dataset(train_summaries, train_codes, language_seqifier, code_seqifier,
                                                   model.l_dim, model.c_dim)
    val_summaries, val_codes = process_dataset(val_summaries, val_codes, language_seqifier, code_seqifier,
                                               model.l_dim, model.c_dim)

    print("Starting training now...")
    train_bvae(model, model_path, train_summaries, train_codes, val_summaries, val_codes)


if __name__ == "__main__":
    main()
