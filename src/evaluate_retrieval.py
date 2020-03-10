import random
from ret_bvae import *
import sys
sys.path.append("../baselines/RET-IR")
import retir


def reciprocal_rank(sorted_indices, golden_idx):
    for i in range(len(sorted_indices)):
        if sorted_indices[i] == golden_idx:
            return 1.0 / (i + 1.0)


def evaluate_retrieval(summaries, codes, baseline='ret_ir', random_sample_size=50, num_samples=20):
    if baseline == 'ret_ir':
        baseline_model = lambda summary, code_snippets: retir.retir(summary, code_snippets, random_sample_size)
    else:
        raise Exception("Invalid baseline specified: %s" % baseline)
    assert len(summaries) == len(codes)
    num_inputs = len(summaries)
    bvae_model = RetBVAE()
    random.seed()
    baseline_reciprocal_ranks = []
    bvae_reciprocal_ranks = []
    for i in range(num_samples):
        rand_idx = random.randrange(0, num_inputs - random_sample_size)
        rand_summaries = summaries[rand_idx: rand_idx + random_sample_size]
        rand_codes = codes[rand_idx: rand_idx + random_sample_size]
        golden_idx = random.randrange(random_sample_size)
        """baseline_sorted_indices = baseline_model(rand_summaries[golden_idx], rand_codes)
        baseline_reciprocal_ranks.append(reciprocal_rank(baseline_sorted_indices, golden_idx))"""
        bvae_sorted_indices = bvae_model.rank_options(rand_summaries[golden_idx], rand_codes)
        bvae_reciprocal_ranks.append(reciprocal_rank(bvae_sorted_indices, golden_idx))

    """baseline_mean_reciprocal_rank = np.mean(baseline_reciprocal_ranks)
    print("Baseline Mean Reciprocal Rank: %s" % baseline_mean_reciprocal_rank)"""

    bvae_mean_reciprocal_rank = np.mean(bvae_reciprocal_ranks)
    print("BVAE Mean Reciprocal Rank: %s" % bvae_mean_reciprocal_rank)


def main():
    summaries, codes = load_iyer_file("../data/iyer_csharp/dev.txt")
    evaluate_retrieval(summaries, codes)


if __name__ == "__main__":
    main()
