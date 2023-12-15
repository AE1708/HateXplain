import random
import numpy as np
from scipy.stats import pearsonr, rankdata
import itertools
from tqdm import tqdm
from scipy.special import comb

def compute_correlation_agreement(hatexplain, best_algorithm, shap_values):
    
    base = []
    features = hatexplain.vectorizer.vocabulary_.keys()
    for sentence, global_rationale in zip(hatexplain.test_sentences, hatexplain.test_rationales):
        individual_base = np.zeros(len(features))
        if any(i == 1 for i in global_rationale):
            for token, rationale in zip(sentence, global_rationale):
                if rationale == 1 and token in hatexplain.vectorizer.vocabulary_:
                    idx = hatexplain.vectorizer.vocabulary_.get(token)
                    individual_base[idx] = 1
        base.append(individual_base) 

    labels = {
        'hatespeech': 0,
        'normal': 1,
        'offensive': 2,
    }

    shaps = []
    for n, i in enumerate(hatexplain.X_test_vect):
        predicted_class = best_algorithm.predict(i)[0]
        shap_idx = labels[predicted_class]
        shaps.append(shap_values[shap_idx][n])


    attrA = shaps
    attrB = base

    min_range = 1
    max_range = len(shaps)

    unique_random_numbers = random.sample(range(min_range, max_range + 1), 100)

    attrA_reduced = []
    attrB_reduced = []

    for n in unique_random_numbers:
        attrA_reduced.append(shaps[n])
        attrB_reduced.append(base[n])

    n_datapoints = len(attrA_reduced)
    n_feat = len(attrA_reduced[0])

    all_feat_ranksA = rankdata(-np.abs(attrA_reduced), method='dense', axis=1)
    all_feat_ranksB = rankdata(-np.abs(attrB_reduced), method='dense', axis=1)

    feat_pairs_w_same_rel_rankings = np.zeros(n_datapoints)

    for i in tqdm(range(n_datapoints), desc="Processing"):
        for feat1, feat2 in itertools.combinations_with_replacement(range(n_feat), 2):
            if feat1 != feat2:
                rel_rankingA = all_feat_ranksA[i, feat1] < all_feat_ranksA[i, feat2]
                rel_rankingB = all_feat_ranksB[i, feat1] < all_feat_ranksB[i, feat2]
                feat_pairs_w_same_rel_rankings[i] += rel_rankingA == rel_rankingB

    pairwise_distr = feat_pairs_w_same_rel_rankings / comb(n_feat, 2)

    return np.mean(pairwise_distr)