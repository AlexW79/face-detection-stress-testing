"""Generate README figures and a summary table from results/*.csv."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
METHODS = ["ViolaJones", "HOGSVM", "CNN"]
LABEL = {"ViolaJones": "Viola-Jones", "HOGSVM": "HOG + SVM", "CNN": "CNN (Tiny YOLOv2)"}
DEGS = ["Original", "Decrease Brightness", "Decrease Contrast", "Salt & Pepper Noise",
        "Gaussian Noise", "Rotation", "Blurred"]
TONES = ["Black", "Brown", "White"]
COLORS = {"Black": "#2b2d42", "Brown": "#c77d3a", "White": "#3a86c7"}

def counts(m, d, t):
    df = pd.read_csv(RES / m / d / f"{t}_Skintone_face_detection_results.csv")
    cols = [c for c in df.columns if c.startswith("Increase")]
    return df[cols].astype(str).isin(["1", "True", "true"]).sum().tolist(), len(df)

rows = []
for m in METHODS:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True)
    for ax, d in zip(axes.flat, DEGS):
        for t in TONES:
            c, n = counts(m, d, t)
            x = [k * 30 for k in range(10)] if d == "Rotation" else list(range(10))
            ax.plot(x, c, "-o", ms=3.5, color=COLORS[t], label=t)
            rows.append(dict(method=m, degradation=d, tone=t, step0=c[0], step9=c[-1], n=n))
        ax.set_title(d, fontsize=11)
        ax.set_xlabel("Rotation (degrees)" if d == "Rotation" else "Degradation step")
        ax.grid(alpha=.3)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Faces detected (of 500)")
    axes[1, 0].set_ylabel("Faces detected (of 500)")
    axes[0, 0].legend(title="Skin tone")
    fig.suptitle(f"{LABEL[m]} — detections per degradation step", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"{m}_degradation.png", dpi=110)
    plt.close(fig)

summary = pd.DataFrame(rows)
summary.to_csv(RES / "summary.csv", index=False)

# Baseline bar chart
base = summary[summary.degradation == "Original"].pivot(index="method", columns="tone", values="step0").loc[METHODS]
ax = base.rename(index=LABEL).plot.bar(color=[COLORS[t] for t in TONES], figsize=(8, 4.5), rot=0)
ax.set_ylabel("Faces detected (of 500)"); ax.set_xlabel("")
ax.set_title("Baseline detections on undegraded images", weight="bold")
ax.grid(axis="y", alpha=.3); ax.legend(title="Skin tone")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "baseline.png", dpi=110)
print(base)
print(summary.pivot_table(index=["method", "degradation"], columns="tone", values=["step0", "step9"]).to_string())
