"""Regenerate the figures of the paper from the result files in results/.

Usage:  python scripts/make_figures.py        (writes figures/*.png)
"""
import math, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.optimize import minimize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7})
TASKS = ["T1", "T1b", "T2", "T3"]
def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig); print("wrote figures/" + name)

# ---------------------------------------------------------------- Fig. 1: corpus and tasks
fig = plt.figure(figsize=(3.35, 2.35)); ax = fig.add_axes([0.20, 0.42, 0.78, 0.55])
n = [1800, 600, 600, 600]
ax.bar(range(4), n, color=["0.85", "0.55", "0.35", "0.20"], edgecolor="black", lw=0.6, width=0.62)
for i, v in enumerate(n): ax.text(i, v + 40, str(v), ha="center", fontsize=6.5)
ax.set_xticks(range(4)); ax.set_ylim(0, 2050); ax.set_ylabel("profiles"); ax.grid(axis="y", lw=0.3, alpha=0.5)
ax.set_xticklabels(["0\nlegitimate", "1\nmanual fake", "10\nChatGPT\n(legit. stats)", "11\nChatGPT\n(fake stats)"], fontsize=6)
roles = {"T1": ["negative", "positive", "not used", "not used"], "T1b": ["neg. (600)", "positive", "not used", "not used"],
         "T2": ["negative", "negative", "positive", "positive"], "T3": ["negative", "positive", "positive", "positive"]}
tx = fig.add_axes([0.20, 0.0, 0.78, 0.22]); tx.set_xlim(-0.5, 3.5); tx.set_ylim(-0.5, 3.5); tx.axis("off")
for r, (t, row) in enumerate(roles.items()):
    tx.text(-0.62, 3 - r, t, ha="right", va="center", fontsize=6.5)
    for c, s in enumerate(row):
        tx.text(c, 3 - r, s, ha="center", va="center", fontsize=5.8, color="0.55" if s == "not used" else "black")
save(fig, "fig1_corpus_tasks.png")

# ---------------------------------------------------------------- Fig. 2: pipeline
fig, ax = plt.subplots(figsize=(3.35, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")
def box(x, y, w, h, t, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", lw=0.7, ec="black", fc=fill))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.2, linespacing=1.25)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.7))
box(2, 106, 46, 10, "Intro, About, Experiences\n(section texts)", "0.92"); box(52, 106, 46, 10, "15 section counts", "0.92")
arr(25, 106, 25, 99); arr(75, 106, 75, 63)
box(4, 88, 42, 10, "frozen RoBERTa,\nmean-pooled  e$_s$"); arr(25, 88, 25, 81)
box(2, 64, 46, 16, "tag handling per section\nSTE: e$_s$ - t$_s$\nproposed: project off t$_s$,\nrenormalise")
box(52, 53, 46, 9, "standardised (training split)"); arr(25, 64, 25, 57)
box(2, 48, 46, 8, "average over sections"); arr(25, 48, 50, 41); arr(75, 53, 50, 41)
box(10, 30, 80, 10, "768 text dims + 15 count dims"); arr(50, 30, 50, 24)
box(4, 11, 92, 12, "HGS over 336 logistic-regression models\nfitness: OWA of fold scores minus gap penalty (5-fold CV, training split)")
arr(50, 11, 50, 5); box(14, 0, 72, 5, "T1 / T1b / T2 / T3 decision", "0.92")
save(fig, "fig2_pipeline.png")

# ---------------------------------------------------------------- Fig. 3: OWA weights
def owa_exp(n, orness):
    a = max(-math.log(max(orness, 1e-6)), 1e-6)
    w = np.exp(-a * np.arange(n) / max(n - 1, 1)); return w / w.sum()
def owa_maxent(n, orness):
    coef = np.array([(n - (i + 1)) / (n - 1) for i in range(n)])
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "eq", "fun": lambda w: coef @ w - orness}]
    res = min((minimize(lambda w: float(np.sum(np.clip(w, 1e-12, 1) * np.log(np.clip(w, 1e-12, 1)))), x0,
                        bounds=[(1e-12, 1)] * n, constraints=cons, method="SLSQP", options={"ftol": 1e-12})
               for x0 in np.random.default_rng(0).dirichlet(np.ones(n), 12)), key=lambda r: r.fun)
    return res.x
fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.55)); lbl = ["weakest", "middle", "strongest"]
for k, o in enumerate([0.10, 0.31, 0.50, 0.90]):
    ax[0].plot(range(3), owa_exp(3, o), marker="os^d"[k], ms=3, lw=0.9, label="orness %.2f" % o, color=str(0.15 + 0.22 * k))
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lbl, fontsize=5.8); ax[0].set_ylabel("weight"); ax[0].set_ylim(0, 0.85)
ax[0].legend(fontsize=5, frameon=False, handlelength=1.4); ax[0].set_title("(a) weight by criterion rank", fontsize=6.4)
we, wm = owa_exp(3, 0.30), owa_maxent(3, 0.30)[::-1]; x = np.arange(3)
ax[1].bar(x - 0.18, we, 0.36, label="exponential, (4)", color="0.30", edgecolor="k", lw=0.4)
ax[1].bar(x + 0.18, wm, 0.36, label="maximum entropy", color="0.72", edgecolor="k", lw=0.4)
ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=5.8); ax[1].set_ylim(0, 0.72)
ax[1].legend(fontsize=5, frameon=False); ax[1].set_title("(b) two constructions at orness 0.30", fontsize=6.4)
for a in ax: a.tick_params(labelsize=5.8); a.grid(axis="y", lw=0.3, alpha=0.4)
fig.tight_layout(pad=0.25); save(fig, "fig3_owa_weights.png")
print("  exponential weights at orness 0.30:", np.round(we, 4), "| maximum entropy:", np.round(wm, 4))

R = pd.read_csv(os.path.join(RES, "results_all.csv")); M = R[R.variant == "main"]
# ---------------------------------------------------------------- Fig. 4: pooled T3 confusion matrix
t3 = M[(M.task == "T3") & (M.strategy == "HGS")]
cm = np.array([[t3.TN.sum(), t3.FP.sum()], [t3.FN.sum(), t3.TP.sum()]])
fig, ax = plt.subplots(figsize=(2.3, 1.9)); ax.imshow(cm, cmap="Greys", vmin=0, vmax=2400)
for i in range(2):
    for j in range(2):
        ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=9, color="white" if cm[i, j] > 1200 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["legitimate", "fake"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["legitimate", "fake"])
ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(pad=0.3); save(fig, "fig4_t3_confusion.png")

# ---------------------------------------------------------------- Fig. 5: search strategies
fig, ax = plt.subplots(figsize=(3.35, 1.95)); off = {"HGS": -0.18, "Random": 0.0, "Grid": 0.18}
mk = {"HGS": ("o", "0.1"), "Random": ("s", "0.55"), "Grid": ("D", "white")}
for i, t in enumerate(TASKS):
    for s, o in off.items():
        g = M[(M.task == t) & (M.strategy == s)]
        if not len(g): continue
        ax.scatter(np.full(len(g), i + o), g.f1_pos, marker=mk[s][0], s=14, facecolor=mk[s][1], edgecolor="black", lw=0.5,
                   label={"HGS": "HGS", "Random": "random search (same budget)", "Grid": "exhaustive grid (seed 42)"}[s] if i == 0 else None, zorder=3)
        if s != "Grid": ax.plot([i + o - 0.07, i + o + 0.07], [g.f1_pos.mean()] * 2, color="black", lw=1.0, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(TASKS); ax.set_ylabel("positive-class F1"); ax.set_ylim(0.945, 1.0)
ax.grid(axis="y", lw=0.3, alpha=0.5); ax.legend(fontsize=5.8, frameon=False, loc="lower right")
fig.tight_layout(pad=0.3); save(fig, "fig5_search_strategies.png")

# ---------------------------------------------------------------- Fig. 6: embedding screen
E = pd.read_csv(os.path.join(RES, "results_embeddings.csv"))
names = {"V1": "plain mean", "V2": "R1 renorm.", "V3": "R2, global mean", "V4": "R2, per-section mean",
         "V5": "tag-direction projection", "V6": "ZCA whitening", "V7": "STE + cosine coherence",
         "V8": "R2 per-section + coherence", "V9": "STE + L1 distances"}
base = E[E.variant == "V0"].set_index(["task", "seed"]).f1_pos
D = np.zeros((9, 4)); W = np.empty((9, 4), dtype=object)
for r, v in enumerate(names):
    for c, t in enumerate(TASKS):
        x = E[(E.variant == v) & (E.task == t)].set_index(["task", "seed"]).f1_pos
        d = (x - base.loc[x.index]).values; D[r, c] = 100 * d.mean(); W[r, c] = "%d/3" % (d > 0).sum()
fig, ax = plt.subplots(figsize=(3.35, 2.55))
im = ax.imshow(np.clip(D, -3, 3), cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
for r in range(9):
    for c in range(4):
        ax.text(c, r, "%+.2f\n%s" % (D[r, c], W[r, c]), ha="center", va="center", fontsize=5.4,
                color="white" if abs(D[r, c]) > 2 else "black", linespacing=1.0)
ax.set_xticks(range(4)); ax.set_xticklabels(TASKS); ax.set_yticks(range(9)); ax.set_yticklabels(list(names.values()), fontsize=6)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=5.5)
cb.set_label("F1 gain over STE (points, clipped at 3)", fontsize=5.5)
fig.tight_layout(pad=0.3); save(fig, "fig6_embedding_screen.png")
