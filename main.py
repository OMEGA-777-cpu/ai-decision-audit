"""
Runs the full pipeline and writes real, reproducible outputs to reports/:
  - results.json          (metrics for all three models)
  - fairness_audit.csv    (subgroup performance table)
  - capability_bench.csv  (latency/scalability table)
  - comparison.png        (accuracy/F1/ROC-AUC bar chart)
  - roc_curves.png
  - capability.png        (latency vs batch size, log-log)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.data import load_data
from src.models import build_models, train_and_evaluate
from src.decision_layer import decide_batch
from src.fairness_audit import subgroup_audit
from src.capability_benchmark import benchmark_inference

OUT = "reports"


def main():
    X_train, X_test, y_train, y_test, feature_names = load_data()

    models = build_models()
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    print("\n=== Model performance ===")
    metrics_out = {}
    for name, r in results.items():
        metrics_out[name] = {k: v for k, v in r.items() if k != "fitted_model"}
        print(name, metrics_out[name])

    # Decision layer demo on the neural network's probabilities
    nn_pipe = results["neural_network"]["fitted_model"]
    p_malignant = 1 - nn_pipe.predict_proba(X_test)[:, 1]
    decisions = decide_batch(p_malignant)
    decision_counts = {d: decisions.count(d) for d in set(decisions)}
    print("\n=== Decision layer output (neural network) ===")
    print(decision_counts)

    # Fairness / subgroup audit (best model by F1)
    best_name = max(results, key=lambda n: results[n]["f1"])
    fairness_df = subgroup_audit(results[best_name]["fitted_model"], X_test, y_test)
    print(f"\n=== Subgroup audit ({best_name}) ===")
    print(fairness_df.to_string(index=False))
    fairness_df.to_csv(f"{OUT}/fairness_audit.csv", index=False)

    # Capability benchmark
    bench_df = benchmark_inference(results, X_test)
    print("\n=== Capability benchmark ===")
    print(bench_df.to_string(index=False))
    bench_df.to_csv(f"{OUT}/capability_bench.csv", index=False)

    metrics_out["decision_layer_counts"] = decision_counts
    metrics_out["best_model_by_f1"] = best_name
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # --- Plot 1: metric comparison bar chart ---
    names = list(results.keys())
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    width = 0.15
    for i, m in enumerate(metrics_to_plot):
        vals = [results[n][m] for n in names]
        ax.bar([xi + i * width for xi in x], vals, width, label=m)
    ax.set_xticks([xi + width * 2 for xi in x])
    ax.set_xticklabels(names)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison: classical ML vs. neural network")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/comparison.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: ROC curves ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        pipe = r["fitted_model"]
        probs = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc_curves.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: capability / scalability ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        sub = bench_df[bench_df["model"] == name]
        ax.plot(sub["batch_size"], sub["ms_per_sample"], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size (log)")
    ax.set_ylabel("ms per sample (log)")
    ax.set_title("Inference latency vs. batch size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/capability.png", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs written to {OUT}/")


if __name__ == "__main__":
    main()
