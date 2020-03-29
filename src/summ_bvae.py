import sys
from bvae import *


assert len(sys.argv) == 2, "Usage: python summ_bvae.py path/to/model/dir/"
model_path = sys.argv[1]

print("Loading model...")
model = BimodalVariationalAutoEncoder(model_path)

while True:
    print()
    code = input("Input Code: ")
    if code == "exit":
        break
    latent = model.codes_to_latent([code])
    summary = model.latent_to_summaries(latent)[0]
    print("Predicted Summary: %s" % summary)
