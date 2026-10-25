
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_PATH = Path("outputs")


# 1. Load similarity scores

def load_scores(hop, label):
    filename = OUTPUT_PATH / f"aves_{hop}hop_combined_test_results.csv"

    with open(filename, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return np.array([
        float(row["max_similarity"])
        for row in rows
        if row["ground_truth"] == label
    ])


known_scores = load_scores(1, "known")
unknown_1hop = load_scores(1, "unknown")
unknown_7hop = load_scores(7, "unknown")


# 2. Display descriptive statistics

def summarize(name, scores):
    print(f"\n{name}")
    print("Images:", len(scores))
    print(f"Mean: {scores.mean():.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Standard deviation: {scores.std():.4f}")
    print(f"Minimum: {scores.min():.4f}")
    print(f"Maximum: {scores.max():.4f}")


summarize("Known images", known_scores)
summarize("1-hop unknown images", unknown_1hop)
summarize("7-hop unknown images", unknown_7hop)



# 3. Plot similarity distributions

from scipy.stats import gaussian_kde

plt.figure(figsize=(10, 5.5))

groups = [
    (known_scores, "Known species", "#8bbbee"),
    (unknown_1hop, "1-hop unknown", "#e18497"),
    (unknown_7hop, "7-hop unknown", "#a8dca7")
]

x = np.linspace(0.18, 0.43, 500)

for scores, label, color in groups:
    kde = gaussian_kde(scores)
    density = kde(x)

    plt.plot(
        x, density,
        label=label,
        color=color,
        linewidth=2.5
    )

    plt.fill_between(
        x, density,
        color=color,
        alpha=0.10
    )

plt.xlabel("Maximum Known-Class Cosine Similarity", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)

plt.title(
    "Zero-Shot CLIP: Known vs. Unknown Similarity Distributions",
    fontsize=14,
    fontweight="bold",
    pad=15
)

plt.legend(frameon=False, fontsize=11)

plt.grid(axis="y", alpha=0.15)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "similarity_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
