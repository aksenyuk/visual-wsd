import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_metrics_align = {
    "model": "ALIGN",
    "accuracy": 0.12805967829668194,
    "f1": 0.22704415512847007,
    "precision": 1.0,
    "recall": 0.12805967829668194,
    "mrr": 0.32633395498225715,
    "time": 5691.2699983119965,
}

results_metrics_blip = {
    "model": "BLIP",
    "accuracy": 0.14523272981583651,
    "f1": 0.25363007192292036,
    "precision": 1.0,
    "recall": 0.14523272981583651,
    "mrr": 0.351030037730636,
    "time": 5908.4345779418945,
}

results_metrics_clip = {
    "model": "CLIP",
    "accuracy": 0.7326132566632994,
    "f1": 0.8456743059604431,
    "precision": 1.0,
    "recall": 0.7326132566632994,
    "mrr": 0.8291554887036277,
    "time": 4890.31906914711,
}

results_bridge_tower = {
    "model": "BridgeTower",
    "accuracy": 0.5027585670992307,
    "f1": 0.669114225140907,
    "precision": 1.0,
    "recall": 0.5027585670992307,
    "mrr": 0.6645167974719611,
    "time": 7538.704027414322,
}

results_group_vit = {
    "model": "GroupViT",
    "accuracy": 0.5027585670992307,
    "f1": 0.669114225140907,
    "precision": 1.0,
    "recall": 0.5027585670992307,
    "mrr": 0.6584506942363031,
    "time": 4864.772887468338,
}

data = [
    results_metrics_align,
    results_metrics_blip,
    results_metrics_clip,
    results_bridge_tower,
    results_group_vit,
]

metrics = list(results_metrics_clip.keys())
metrics.remove("model")

sns.set_style("whitegrid")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle("Model Performance Comparison", fontsize=16)

for i, metric in enumerate(metrics):
    ax = axes[i // 3, i % 3]
    sns.barplot(x="model", y=metric, data=pd.DataFrame(data), ax=ax)
    ax.set_title(metric.capitalize())
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.capitalize())

    if metric == "accuracy":
        ax.axhline(y=0.10, color="black", linestyle="--", linewidth=3)
        ax.text(1, 0.12, "Random Guessing", color="black", fontsize=15, ha="center")


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("metric_performance_plots/metric_performance.png")
