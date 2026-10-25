
import csv
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("outputs")

# 1. Load similarity scores

def load_scores(filename, label):
    with open(filename, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return np.array([
        float(row["max_similarity"])
        for row in rows
        if row["ground_truth"] == label
    ])


# 2. Calibrate threshold using development data

development_file = (
    OUTPUT_PATH / "aves_1hop_combined_development_results.csv"
)

known_dev_scores = load_scores(development_file, "known")

# Accept approximately 95% of known development images
threshold = np.percentile(known_dev_scores, 5)

print(f"Calibrated threshold: {threshold:.6f}")

development_acceptance = np.mean(
    known_dev_scores >= threshold
)

print(f"Development known acceptance: {development_acceptance:.2%}")


# 3. Apply the SAME threshold across all seven hops

print("\n--- FIXED-THRESHOLD EVALUATION ---")

for hop in range(1, 8):
    filename = (
        OUTPUT_PATH / f"aves_{hop}hop_combined_test_results.csv"
    )

    known_scores = load_scores(filename, "known")
    unknown_scores = load_scores(filename, "unknown")

    known_acceptance = np.mean(
        known_scores >= threshold
    )

    false_acceptance = np.mean(
        unknown_scores >= threshold
    )

    print(
        f"{hop}-hop | "
        f"Known acceptance: {known_acceptance:.2%} | "
        f"Unknown false acceptance: {false_acceptance:.2%}"
    )

# 4. Plot false acceptance across hops

import matplotlib.pyplot as plt

hops = list(range(1, 8))
false_acceptance_rates = []

for hop in hops:
    filename = OUTPUT_PATH / f"aves_{hop}hop_combined_test_results.csv"
    unknown_scores = load_scores(filename, "unknown")

    rate = np.mean(unknown_scores >= threshold) * 100
    false_acceptance_rates.append(rate)

plt.figure(figsize=(9, 5))

plt.plot(
    hops,
    false_acceptance_rates,
    color="#a8dca7",
    marker="o",
    linewidth=2.5,
    markersize=7
)

plt.xlabel("Taxonomic Distance (Hops)", fontsize=12)
plt.ylabel("Unknown False Acceptance Rate (%)", fontsize=12)
plt.title(
    "False Acceptance Rate Across Taxonomic Distances",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.xticks(hops)
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.15)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(OUTPUT_PATH / "false_acceptance_by_hop.png", dpi=300)
plt.savefig(OUTPUT_PATH / "false_acceptance_by_hop.pdf")

plt.show()
