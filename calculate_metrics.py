
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

OUTPUT_PATH = Path("outputs")

def evaluate_prompt(prompt_type):
    filename = OUTPUT_PATH / f"aves_1hop_{prompt_type}_development_results.csv"

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

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fpr95 = fpr[tpr >= 0.95].min()

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "fpr95": fpr95,
        "known_mean": scores[labels == 1].mean(),
        "unknown_mean": scores[labels == 0].mean()
    }

for prompt_type in ["scientific", "common", "combined"]:
    result = evaluate_prompt(prompt_type)

    print(f"\n--- {prompt_type.upper()} PROMPTS ---")
    print(f"Known accuracy: {result['accuracy']:.2%}")
    print(f"AUROC: {result['auroc']:.4f}")
    print(f"FPR95: {result['fpr95']:.4f}")
    print(f"Mean known similarity: {result['known_mean']:.4f}")
    print(f"Mean unknown similarity: {result['unknown_mean']:.4f}")


# Held-out Aves 1-hop evaluation

import csv
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

with open(
    "outputs/aves_1hop_combined_test_results.csv",
    newline="",
    encoding="utf-8"
) as f:
    test_results = list(csv.DictReader(f))

labels = np.array([
    1 if row["ground_truth"] == "known" else 0
    for row in test_results
])

scores = np.array([
    float(row["max_similarity"])
    for row in test_results
])

known_results = [
    row for row in test_results
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

print("\n--- HELD-OUT AVES 1-HOP ---")
print("Images evaluated:", len(test_results))
print("Known images:", len(known_results))
print("Unknown images:", sum(labels == 0))

print(f"Known accuracy: {accuracy:.2%}")
print(f"AUROC: {auroc:.4f}")
print(f"FPR95: {fpr95:.4f}")
print(f"Mean known similarity: {scores[labels == 1].mean():.4f}")
print(f"Mean unknown similarity: {scores[labels == 0].mean():.4f}")