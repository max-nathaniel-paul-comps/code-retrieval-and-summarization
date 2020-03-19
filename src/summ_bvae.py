import sys
from bvae import *
from seqifier import Seqifier


def main():
    assert len(sys.argv) == 2, "Usage: python summ_bvae.py path/to/model/dir/"
    model_path = sys.argv[1]

    print("Loading seqifiers, which are responsible for turning texts into sequences of integers...")
    with open(model_path + "seqifiers_description.json") as seq_desc_json:
        seqifiers_description = json.load(seq_desc_json)
    language_seqifier = Seqifier(seqifiers_description['language_seq_type'],
                                 model_path + seqifiers_description['language_seq_path'])
    code_seqifier = Seqifier(seqifiers_description['source_code_seq_type'],
                             model_path + seqifiers_description['source_code_seq_path'])

    print("Loading model...")
    model = BimodalVariationalAutoEncoder(model_path, language_seqifier, code_seqifier)
    model.compile()
    model.load_weights(model_path + "model_checkpoint.ckpt")

    while True:
        print()
        code = input("Input Code: ")
        if code == "exit":
            break
        code_seq = code_seqifier.seqify_texts([code])
        code_emb = model.source_code_embedding(code_seq)
        latent = model.source_code_encoder(code_emb)
        summary_seq = model.language_decoder(latent.mean())
        summary = language_seqifier.de_seqify_texts(summary_seq)[0]
        print("Predicted Summary: %s" % summary)


if __name__ == "__main__":
    main()
