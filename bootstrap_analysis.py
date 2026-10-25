
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

OUTPUT_PATH = Path("outputs")
N_BOOTSTRAP = 1000
SEED = 42

# Same fixed threshold used in the previous analysis
def load_scores(filename, label):
    with open(filename, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return np.array([
        float(row["max_similarity"])
        for row in rows
        if row["ground_truth"] == label
    ])

development_file = OUTPUT_PATH / "aves_1hop_combined_development_results.csv"
threshold = np.percentile(load_scores(development_file, "known"), 5)

# Load scores grouped by species
def load_species_scores(hop):
    filename = OUTPUT_PATH / f"aves_{hop}hop_combined_test_results.csv"

    known = defaultdict(list)
    unknown = defaultdict(list)

    with open(filename, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            category = int(row["actual_category"])
            score = float(row["max_similarity"])

            if row["ground_truth"] == "known":
                known[category].append(score)
            else:
                unknown[category].append(score)

    return known, unknown


# Resample species with replacement
def sample_species(group, rng):
    species_ids = list(group.keys())

    selected = rng.choice(
        species_ids,
        size=len(species_ids),
        replace=True
    )

    return np.concatenate([
        np.array(group[species_id])
        for species_id in selected
    ])


def calculate_ci(hop):
    known, unknown = load_species_scores(hop)

    known_scores = np.concatenate(list(known.values()))
    unknown_scores = np.concatenate(list(unknown.values()))

    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores))
    ])

    scores = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(labels, scores)
    false_acceptance = np.mean(unknown_scores >= threshold)

    rng = np.random.default_rng(SEED)

    bootstrap_aurocs = []
    bootstrap_false_acceptance = []

    for _ in range(N_BOOTSTRAP):
        sampled_known = sample_species(known, rng)
        sampled_unknown = sample_species(unknown, rng)

        sampled_labels = np.concatenate([
            np.ones(len(sampled_known)),
            np.zeros(len(sampled_unknown))
        ])

        sampled_scores = np.concatenate([
            sampled_known,
            sampled_unknown
        ])

        bootstrap_aurocs.append(
            roc_auc_score(sampled_labels, sampled_scores)
        )

        bootstrap_false_acceptance.append(
            np.mean(sampled_unknown >= threshold)
        )

    auroc_ci = np.percentile(bootstrap_aurocs, [2.5, 97.5])
    far_ci = np.percentile(bootstrap_false_acceptance, [2.5, 97.5])

    return auroc, auroc_ci, false_acceptance, far_ci


print("95% Species-Level Bootstrap Confidence Intervals")
print("Bootstrap repetitions:", N_BOOTSTRAP)

for hop in range(1, 8):
    auroc, auroc_ci, far, far_ci = calculate_ci(hop)

    print(
        f"\n{hop}-hop | "
        f"AUROC: {auroc:.4f} "
        f"[{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}] | "
        f"False acceptance: {far:.2%} "
        f"[{far_ci[0]:.2%}, {far_ci[1]:.2%}]"
    )

# 4. Compare 1-hop and 7-hop directly

known_1, unknown_1 = load_species_scores(1)
known_7, unknown_7 = load_species_scores(7)

assert known_1 == known_7

known_scores = np.concatenate(list(known_1.values()))
unknown_1_scores = np.concatenate(list(unknown_1.values()))
unknown_7_scores = np.concatenate(list(unknown_7.values()))


def calculate_auroc(known_scores, unknown_scores):
    labels = np.concatenate([
        np.ones(len(known_scores)),
        np.zeros(len(unknown_scores))
    ])

    scores = np.concatenate([
        known_scores,
        unknown_scores
    ])

    return roc_auc_score(labels, scores)


# Calculate observed differences

auroc_difference = (
    calculate_auroc(known_scores, unknown_7_scores)
    - calculate_auroc(known_scores, unknown_1_scores)
)

far_difference = (
    np.mean(unknown_1_scores >= threshold)
    - np.mean(unknown_7_scores >= threshold)
)


# Paired bootstrap: reuse sampled known species

rng = np.random.default_rng(SEED)

auroc_differences = []
far_differences = []

for _ in range(N_BOOTSTRAP):
    sampled_known = sample_species(known_1, rng)
    sampled_unknown_1 = sample_species(unknown_1, rng)
    sampled_unknown_7 = sample_species(unknown_7, rng)

    auroc_1 = calculate_auroc(sampled_known, sampled_unknown_1)
    auroc_7 = calculate_auroc(sampled_known, sampled_unknown_7)

    auroc_differences.append(auroc_7 - auroc_1)

    far_differences.append(
        np.mean(sampled_unknown_1 >= threshold)
        - np.mean(sampled_unknown_7 >= threshold)
    )


# Calculate confidence intervals for differences

auroc_diff_ci = np.percentile(
    auroc_differences, [2.5, 97.5]
)

far_diff_ci = np.percentile(
    far_differences, [2.5, 97.5]
)

print("\n--- 1-HOP VS 7-HOP COMPARISON ---")

print(f"AUROC difference: {auroc_difference:.4f}")
print(
    f"95% CI: "
    f"[{auroc_diff_ci[0]:.4f}, {auroc_diff_ci[1]:.4f}]"
)

print(f"\nFalse acceptance difference: {far_difference:.2%}")
print(
    f"95% CI: "
    f"[{far_diff_ci[0]:.2%}, {far_diff_ci[1]:.2%}]"
)
