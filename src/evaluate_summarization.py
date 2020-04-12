import sys
import nltk
import numpy as np
import text_data_utils as tdu
from bvae import BimodalVariationalAutoEncoder


nltk.download('wordnet')

assert len(sys.argv) == 2, "Usage: python evaluate_summarization.py path/to/bvae/model/dir/"
bvae_model_path = sys.argv[1]

dataset = tdu.load_iyer_dataset("../data/iyer_csharp/dev.txt",
                                alternate_summaries_filename="../data/iyer_csharp/dev_alternate_summaries.txt")

bvae_model = BimodalVariationalAutoEncoder(bvae_model_path)

codes = [ex[1] for ex in dataset]
dataset_latent = bvae_model.codes_to_latent(codes)
dataset_summarized = bvae_model.latent_to_summaries(dataset_latent)

true_summaries = [[ex[0]] + ex[2] for ex in dataset]

meteors = []
for i in range(len(dataset)):
    meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], dataset_summarized[i])
    meteors.append(meteor)
    print("Code: %s" % codes[i])
    print("True Summaries: %s" % true_summaries[i])
    print("Predicted Summary: %s" % dataset_summarized[i])
    print("Summary METEOR score: %s\n" % meteor)

average_meteor = np.mean(meteors)
print("\nAverage METEOR score: %s" % average_meteor)

corpus_bleu = nltk.translate.bleu_score.corpus_bleu(true_summaries, dataset_summarized)
print("Corpus BLEU score: %s" % corpus_bleu)
