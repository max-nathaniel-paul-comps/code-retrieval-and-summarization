import sys
import nltk
import numpy as np
import text_data_utils as tdu
import tqdm
import random
from bvae import BimodalVariationalAutoEncoder
from transformer import CodeSummarizationTransformer


nltk.download('wordnet')

assert len(sys.argv) == 4, "Usage: python evaluate_summarization.py prog_lang path/to/bvae/model/dir/ path/to/transformer"
prog_lang = sys.argv[1]
bvae_model_path = sys.argv[2]
transformer_model_path = sys.argv[3]

if prog_lang == "csharp":
    dataset = tdu.load_iyer_dataset("../data/iyer_csharp/dev.txt",
                                    alternate_summaries_filename="../data/iyer_csharp/dev_alternate_summaries.txt")
elif prog_lang == "python":
    _, _, dataset = tdu.load_edinburgh_dataset("../data/edinburgh_python")
elif prog_lang == "java":
    dataset = tdu.load_json_dataset("../data/leclair_java/test.json")
    dataset = random.sample(dataset, 300)
else:
    raise Exception("lmao")

if bvae_model_path != "no_bvae":
    bvae_model = BimodalVariationalAutoEncoder(bvae_model_path)
if transformer_model_path != "no_transformer":
    transformer_model = CodeSummarizationTransformer(transformer_model_path)

codes = [ex[1] for ex in dataset]

if bvae_model_path != "no_bvae":
    dataset_latent_bvae = bvae_model.codes_to_latent(codes)
    dataset_summarized_bvae = bvae_model.latent_to_summaries(dataset_latent_bvae)
if transformer_model_path != "no_transformer":
    dataset_summarized_transformer = []
    for i in tqdm.trange(len(codes)):
        dataset_summarized_transformer.append(transformer_model.transformer.translate(codes[i], print_output=False))

if prog_lang == "python":
    true_summaries = [[ex[0]] + ex[2] for ex in dataset]
else:
    true_summaries = [[ex[0]] for ex in dataset]

bvae_meteors = []
transformer_meteors = []
for i in range(len(dataset)):
    print("Code: %s" % codes[i])
    print("True Summaries: %s" % true_summaries[i])
    if bvae_model_path != "no_bvae":
        bvae_meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], dataset_summarized_bvae[i])
        bvae_meteors.append(bvae_meteor)
        print("BVAE Predicted Summary: %s" % dataset_summarized_bvae[i])
        print("BVAE Summary METEOR score: %s" % bvae_meteor)
    if transformer_model_path != "no_transformer":
        transformer_meteor = nltk.translate.meteor_score.meteor_score(true_summaries[i], dataset_summarized_transformer[i])
        transformer_meteors.append(transformer_meteor)
        print("Transformer Predicted Summary: %s" % dataset_summarized_transformer[i])
        print("Transformer METEOR score: %s\n" % transformer_meteor)

if bvae_model_path != "no_bvae":
    average_bvae_meteor = np.mean(bvae_meteors)
    print("\nAverage BVAE METEOR score: %s" % average_bvae_meteor)
    bvae_corpus_bleu = nltk.translate.bleu_score.corpus_bleu(true_summaries, dataset_summarized_bvae)
    print("BVAE Corpus BLEU score: %s" % bvae_corpus_bleu)

if transformer_model_path != "no_transformer":
    average_transformer_meteor = np.mean(transformer_meteors)
    print("Average Transformer METEOR score: %s\n" % average_transformer_meteor)
    transformer_corpus_bleu = nltk.translate.bleu_score.corpus_bleu(true_summaries, dataset_summarized_transformer)
    print("Transformer Corpus BLEU score: %s" % transformer_corpus_bleu)
