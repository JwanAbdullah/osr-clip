
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

OUTPUT_PATH = Path("outputs")


# 1. Calculate evaluation metrics

def calculate_metrics(filename):
    with open(filename, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    labels = np.array([
        1 if row["ground_truth"] == "known" else 0
        for row in results
    ])

    scores = np.array([
        float(row["max_similarity"])
        for row in results
    ])

    known_results = [
        row for row in results
        if row["ground_truth"] == "known"
    ]

    correct = sum(
        row["actual_category"] == row["predicted_category"]
        for row in known_results
    )

    accuracy = correct / len(known_results)
    auroc = roc_auc_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    fpr95 = fpr[tpr >= 0.95].min()

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "fpr95": fpr95,
        "known_mean": scores[labels == 1].mean(),
        "unknown_mean": scores[labels == 0].mean(),
        "known_count": int(sum(labels == 1)),
        "unknown_count": int(sum(labels == 0))
    }


# 2. Compare prompt types on development set

print("\n--- DEVELOPMENT PROMPT COMPARISON ---")

for prompt_type in ["scientific", "common", "combined"]:
    filename = OUTPUT_PATH / f"aves_1hop_{prompt_type}_development_results.csv"

    result = calculate_metrics(filename)

    print(f"\n{prompt_type.upper()} PROMPTS")
    print(f"Known accuracy: {result['accuracy']:.2%}")
    print(f"AUROC: {result['auroc']:.4f}")
    print(f"FPR95: {result['fpr95']:.4f}")
    print(f"Mean known similarity: {result['known_mean']:.4f}")
    print(f"Mean unknown similarity: {result['unknown_mean']:.4f}")


# 3. Compare held-out results across all seven hops

print("\n--- HELD-OUT AVES 1-7 HOP EVALUATION ---")

for hop in range(1, 8):
    filename = OUTPUT_PATH / f"aves_{hop}hop_combined_test_results.csv"

    if not filename.exists():
        continue

    result = calculate_metrics(filename)

    print(
        f"{hop}-hop | "
        f"Accuracy: {result['accuracy']:.2%} | "
        f"AUROC: {result['auroc']:.4f} | "
        f"FPR95: {result['fpr95']:.4f} | "
        f"Known: {result['known_count']} | "
        f"Unknown: {result['unknown_count']}"
    )



# 4. Plot AUROC and FPR95 across taxonomic hops

import matplotlib.pyplot as plt

hops = []
auroc_values = []
fpr95_values = []

for hop in range(1, 8):
    filename = OUTPUT_PATH / f"aves_{hop}hop_combined_test_results.csv"

    result = calculate_metrics(filename)

    hops.append(hop)
    auroc_values.append(result["auroc"])
    fpr95_values.append(result["fpr95"] * 100)


# AUROC graph

plt.figure(figsize=(9, 5))

plt.plot(
    hops, auroc_values,
    color="#80B5EF",
    marker="o",
    linewidth=2.5,
    markersize=7
)

plt.xlabel("Taxonomic Distance (Hops)", fontsize=12)
plt.ylabel("AUROC", fontsize=12)
plt.title("AUROC Across Taxonomic Distances", fontsize=14, fontweight="bold")

plt.xticks(hops)
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.15)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.axhline(
    y=0.5,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    alpha=0.6,
    label="Random-ranking reference"
)

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "auroc_by_hop.png", dpi=300)
plt.savefig(OUTPUT_PATH / "auroc_by_hop.pdf")
plt.show()


# FPR95 graph

plt.figure(figsize=(9, 5))

plt.plot(
    hops, fpr95_values,
    color="#E87C96",
    marker="o",
    linewidth=2.5,
    markersize=7
)

plt.xlabel("Taxonomic Distance (Hops)", fontsize=12)
plt.ylabel("FPR95 (%)", fontsize=12)
plt.title("FPR95 Across Taxonomic Distances", fontsize=14, fontweight="bold")

plt.xticks(hops)
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.15)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "fpr95_by_hop.png", dpi=300)
plt.savefig(OUTPUT_PATH / "fpr95_by_hop.pdf")
plt.show()
