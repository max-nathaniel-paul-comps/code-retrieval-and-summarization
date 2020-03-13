import random
import numpy as np
from tqdm import tqdm
from bvae import *
import sys
sys.path.append("../baselines/RET-IR")
import retir


def reciprocal_rank(sorted_indices, golden_idx):
    for i in range(len(sorted_indices)):
        if sorted_indices[i] == golden_idx:
            return 1.0 / (i + 1.0)


def evaluate_retrieval(summaries, codes, bvae_model_path,
                       baseline='random', random_sample_size=50, num_samples=1000):
    if baseline == 'ret_ir':
        def baseline_model(summary, candidate_summaries, candidate_codes):
            summary = retir.shuffleQuery(summary, 0.2)
            # Calling the version for pretokenized because ret-ir performs better at character-level for some reason
            return retir.retir_pt(summary, candidate_summaries, random_sample_size)
    elif baseline == 'random':
        def baseline_model(summary, candidate_summaries, candidate_codes):
            return random.sample(range(len(candidate_summaries)), len(candidate_summaries))
    else:
        raise Exception("Invalid baseline specified: %s" % baseline)

    assert len(summaries) == len(codes)
    num_inputs = len(summaries)

    bvae_model = load_or_create_model(bvae_model_path)

    language_seqifier = load_or_create_seqifier(bvae_model_path + "language_seqifier.json", bvae_model.l_vocab_size)
    code_seqifier = load_or_create_seqifier(bvae_model_path + "code_seqifier.json", bvae_model.c_vocab_size)

    random.seed()
    baseline_reciprocal_ranks = []
    bvae_reciprocal_ranks = []
    for _ in tqdm(range(num_samples)):
        rand_idx = random.randrange(0, num_inputs - random_sample_size)
        rand_summaries = summaries[rand_idx: rand_idx + random_sample_size]
        rand_codes = codes[rand_idx: rand_idx + random_sample_size]

        golden_idx = random.randrange(random_sample_size)

        baseline_sorted_indices = baseline_model(rand_summaries[golden_idx], rand_summaries, rand_codes)
        baseline_reciprocal_ranks.append(reciprocal_rank(baseline_sorted_indices, golden_idx))

        retriever = RetBVAE(bvae_model, rand_codes, language_seqifier, code_seqifier)
        bvae_sorted_indices = retriever.rank_options(rand_summaries[golden_idx])
        bvae_reciprocal_ranks.append(reciprocal_rank(bvae_sorted_indices, golden_idx))

    baseline_mean_reciprocal_rank = np.mean(baseline_reciprocal_ranks)
    print("Baseline Mean Reciprocal Rank: %s" % baseline_mean_reciprocal_rank)

    bvae_mean_reciprocal_rank = np.mean(bvae_reciprocal_ranks)
    print("BVAE Mean Reciprocal Rank: %s" % bvae_mean_reciprocal_rank)


def main():
    summaries, codes = load_iyer_file("../data/iyer_csharp/dev.txt")
    evaluate_retrieval(summaries, codes, "../models/r8/")


if __name__ == "__main__":
    main()
