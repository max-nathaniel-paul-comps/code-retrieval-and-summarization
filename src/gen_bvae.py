import sys
from bvae import *


assert len(sys.argv) == 2, "Usage: python gen_bvae.py path/to/model/dir/"
model_path = sys.argv[1]

print("Loading model...")
model = BimodalVariationalAutoEncoder(model_path)

while True:
    print()
    summary = input("Input Summary: ")
    if summary == "exit":
        break
    latent = model.summaries_to_latent([summary])
    code = model.latent_to_codes(latent)[0]
    print("Predicted Summary: %s" % code)
