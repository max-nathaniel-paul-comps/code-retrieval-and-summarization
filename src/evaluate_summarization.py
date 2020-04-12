import sys
import nltk
import numpy as np
import text_data_utils as tdu
from bvae import BimodalVariationalAutoEncoder
from transformer import CodeSummarizationTransformer


nltk.download('wordnet')

assert len(sys.argv) == 3, "Usage: python evaluate_summarization.py path/to/bvae/model/dir/ path/to/transformer"
bvae_model_path = sys.argv[1]
transformer_model_path = sys.argv[2]

dataset = tdu.load_iyer_dataset("../data/iyer_csharp/dev.txt",
                                alternate_summaries_filename="../data/iyer_csharp/dev_alternate_summaries.txt")

bvae_model = BimodalVariationalAutoEncoder(bvae_model_path)
transformer_model = CodeSummarizationTransformer(transformer_model_path)

codes = [ex[1] for ex in dataset]

dataset_latent_bvae = bvae_model.codes_to_latent(codes)
dataset_summarized_bvae = bvae_model.latent_to_summaries(dataset_latent_bvae)

dataset_summarized_transformer = [transformer_model.transformer.translate(code, print_output=False) for code in codes]

true_summaries = [[ex[0]] + ex[2] for ex in dataset]

bvae_meteors = []
transformer_meteors = []
for i in range(len(dataset)):
    bvae_meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], dataset_summarized_bvae[i])
    bvae_meteors.append(bvae_meteor)
    transformer_meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], dataset_summarized_transformer[i])
    transformer_meteors.append(transformer_meteor)
    print("Code: %s" % codes[i])
    print("True Summaries: %s" % true_summaries[i])
    print("BVAE Predicted Summary: %s" % dataset_summarized_bvae[i])
    print("BVAE Summary METEOR score: %s" % bvae_meteor)
    print("Transformer Predicted Summary: %s" % dataset_summarized_transformer[i])
    print("Transformer METEOR score: %s\n" % transformer_meteor)

average_bvae_meteor = np.mean(bvae_meteors)
print("\nAverage BVAE METEOR score: %s" % average_bvae_meteor)
average_transformer_meteor = np.mean(transformer_meteors)
print("Average Transformer METEOR score: %s\n" % average_transformer_meteor)

bvae_corpus_bleu = nltk.translate.bleu_score.corpus_bleu(true_summaries, dataset_summarized_bvae)
print("BVAE Corpus BLEU score: %s" % bvae_corpus_bleu)
transformer_corpus_bleu = nltk.translate.bleu_score.corpus_bleu(true_summaries, dataset_summarized_transformer)
print("Transformer Corpus BLEU score: %s" % transformer_corpus_bleu)
