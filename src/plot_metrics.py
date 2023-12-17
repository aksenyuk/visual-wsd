import matplotlib.pyplot as plt

resuls_metrics_align = {
    "model": "ALIGN",
    "accuracy": 0.12805967829668194,
    "f1": 0.22704415512847007,
    "precision": 1.0,
    "recall": 0.12805967829668194,
    "mrr": 0.32633395498225715,
    "time": 5691.2699983119965,
}

resuls_metrics_blip = {
    "model": "BLIP",
    "accuracy": 0.14523272981583651,
    "f1": 0.25363007192292036,
    "precision": 1.0,
    "recall": 0.14523272981583651,
    "mrr": 0.351030037730636,
    "time": 5908.4345779418945,
}

resuls_metrics_clip = {
    "model": "CLIP",
    "accuracy": 0.7326132566632994,
    "f1": 0.8456743059604431,
    "precision": 1.0,
    "recall": 0.7326132566632994,
    "mrr": 0.8291554887036277,
    "time": 4890.31906914711,
}

data = [resuls_metrics_align, resuls_metrics_blip, resuls_metrics_clip]

metrics = list(resuls_metrics_clip.keys())
metrics.remove("model")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle("Model Performance Comparison", fontsize=16)

for i, metric in enumerate(metrics):
    ax = axes[i // 3, i % 3]
    ax.bar([d["model"] for d in data], [d[metric] for d in data])
    ax.set_title(metric.capitalize())
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.capitalize())

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("metric_performance_plots/metric_performance.png")
