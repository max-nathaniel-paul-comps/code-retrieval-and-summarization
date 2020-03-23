import sys
from bvae import *
from seqifier import Seqifier


def main():
    assert len(sys.argv) == 2, "Usage: python gen_bvae.py path/to/model/dir/"
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

    while True:
        print()
        query = input("Input Summary: ")
        if query == "exit":
            break
        query_seq = language_seqifier.seqify_texts([query])
        latent = model.language_encoder(query_seq)
        code_seq = model.source_code_decoder(latent.mean())
        code = code_seqifier.de_seqify_texts(code_seq)[0]
        print("Predicted Source Code: %s" % code)


if __name__ == "__main__":
    main()
