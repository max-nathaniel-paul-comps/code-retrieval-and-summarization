import sys
from bvae import *
from text_data_utils import *
from seqifier import *


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

    print("Preparing datasets for training...")
    train_summaries = language_seqifier.seqify_texts(train_summaries)
    train_codes = code_seqifier.seqify_texts(train_codes)
    val_summaries = language_seqifier.seqify_texts(val_summaries)
    val_codes = code_seqifier.seqify_texts(val_codes)

    print("Starting training now...")
    model.train(train_summaries, train_codes, val_summaries, val_codes)


if __name__ == "__main__":
    main()
