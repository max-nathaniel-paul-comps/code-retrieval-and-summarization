import random
from ret_bvae import *


def evaluate_retrieval(summaries, codes, random_sample_size=50, num_samples=80):
    assert len(summaries) == len(codes)
    num_inputs = len(summaries)
    bvae_model = RetBVAE()
    random.seed()
    reciprocal_ranks = []
    for i in range(num_samples):
        rand_idx = random.randrange(0, num_inputs - random_sample_size)
        rand_summaries = summaries[rand_idx: rand_idx + random_sample_size]
        rand_codes = codes[rand_idx: rand_idx + random_sample_size]
        golden_idx = random.randrange(random_sample_size)
        sorted_indices = bvae_model.rank_options(rand_summaries[golden_idx], rand_codes)
        for j in range(len(sorted_indices)):
            if sorted_indices[j] == golden_idx:
                reciprocal_ranks.append(1.0 / (j + 1.0))
    mean_reciprocal_rank = np.mean(reciprocal_ranks)
    return mean_reciprocal_rank


def main():
    summaries, codes = load_iyer_file("../data/iyer_csharp/test.txt")
    mrr = evaluate_retrieval(summaries, codes)
    print("BVAE Mean Reciprocal Rank: %s" % mrr)


if __name__ == "__main__":
    main()
