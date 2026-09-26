"""Regenerate the figures of the paper from the result files in results/.

Figures are drawn at their printed size for the AECE two-column format: text is never smaller than
8 pt and every figure fits one 3.35-inch column, so it is inserted at 100 percent. Figure numbers
follow the paper.

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
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "mathtext.fontset": "stix"})
COLW, MINPT = 3.35, 8.0
R = pd.read_csv(os.path.join(RES, "results_all.csv")); M = R[R.variant == "main"]
E = pd.read_csv(os.path.join(RES, "results_embeddings.csv"))
TASKS = ["T1", "T1b", "T2", "T3"]
import json
B = pd.read_csv(os.path.join(RES, "results_bestemb.csv")); P = pd.read_csv(os.path.join(RES, "results_peerj.csv"))
HG = M[M.strategy == "HGS"]   # HGS rows (H is used below for a figure height)

def check_fonts(fig, name):
    small = [(t.get_text()[:30], t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible() and t.get_fontsize() < MINPT]
    assert not small, (name, small)

def save(fig, name):
    check_fonts(fig, name)
    path = os.path.join(FIG, name + ".png")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    from PIL import Image
    w = Image.open(path).size[0] / 600
    assert w <= COLW + 0.02, (name, w)
    plt.close(fig); print("wrote figures/%s.png (%.2f in wide)" % (name, w))

# ------------------------------------------------------------------ Fig. 1  corpus and tasks
PAL = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]      # light/dark pairs: distinct in grayscale
fig = plt.figure(figsize=(COLW, 2.75))
ax = fig.add_axes([0.17, 0.50, 0.81, 0.47])
n = [1800, 600, 600, 600]
bars = ax.bar(range(4), n, color=PAL, edgecolor="black", lw=0.6, width=0.62)
for b_, h in zip(bars, ["", "//", "\\\\", ".."]): b_.set_hatch(h)
for i, v in enumerate(n): ax.text(i, v + 45, str(v), ha="center", fontsize=8)
ax.set_xticks(range(4)); ax.set_ylim(0, 2100); ax.set_ylabel("Profiles")
ax.set_xticklabels(["0\nlegitimate", "1\nmanual fake", "10\nChatGPT,\nlegit. stats", "11\nChatGPT,\nfake stats"])
ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_axisbelow(True)
roles = {"T1": ["neg.", "pos.", "-", "-"], "T1b": ["neg. (600)", "pos.", "-", "-"],
         "T2": ["neg.", "neg.", "pos.", "pos."], "T3": ["neg.", "pos.", "pos.", "pos."],
         "T4": ["class", "class", "class", "class"]}
tx = fig.add_axes([0.17, 0.0, 0.81, 0.29]); tx.set_xlim(-0.5, 3.5); tx.set_ylim(-0.5, 4.5); tx.axis("off")
for r, (t, row) in enumerate(roles.items()):
    tx.text(-0.72, 4 - r, t, ha="right", va="center", fontsize=8, fontweight="bold")
    for c, s in enumerate(row):
        tx.text(c, 4 - r, s, ha="center", va="center", fontsize=8,
                color="#1f4e79" if s.startswith("pos") else ("#7f7f7f" if s == "-" else "black"))
save(fig, "fig1_corpus_tasks")

# ------------------------------------------------------------------ Fig. 2  pipeline
C_IN, C_ENC, C_TAG, C_FEAT, C_SRCH, C_OUT = "#dbe9f6", "#c6dbef", "#fdd0a2", "#e5f5e0", "#c7e9c0", "#e0d9f0"
W = COLW - 0.08; GAP = 0.13; LINE = 0.145; PADV = 0.07
def hgt(nlines): return nlines * LINE + 2 * PADV
rows = [("in", 2), ("tag", 2), ("avg", 1), ("cat", 2), ("hgs", 3), ("out", 1)]
H = sum(hgt(k) for _, k in rows) + GAP * (len(rows) - 1) + 0.08
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
Y = {}; y = H - 0.04
for key, k in rows:
    Y[key] = (y - hgt(k), hgt(k)); y -= hgt(k) + GAP
BOXES = []
def box(x, key, w, text, fc):
    y0, h = Y[key]
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                lw=0.8, ec="#333333", fc=fc))
    t = ax.text(x + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=8, linespacing=1.15)
    BOXES.append((t, (x, y0, w, h)))
def down(x, key_from, key_to):
    ax.add_patch(FancyArrowPatch((x, Y[key_from][0]), (x, Y[key_to][0] + Y[key_to][1]),
                                 arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333"))
M_ = 0.06; HALF = (W - 2 * M_ - 0.22) / 2
xa, xb = M_, M_ + HALF + 0.22                      # two half-width columns
box(xa, "in", HALF, "Section texts: Intro,\nAbout, Experiences", C_IN)
box(xb, "in", HALF, "Frozen RoBERTa, mean\npooling: $\\mathbf{e}_s$", C_ENC)
yin = Y["in"][0] + Y["in"][1] / 2
ax.add_patch(FancyArrowPatch((xa + HALF, yin), (xb, yin), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333"))
box(M_, "tag", W - 2 * M_, "Tag handling per section.  STE: $\\mathbf{e}_s-\\mathbf{t}_s$\nProposed: project $\\mathbf{e}_s$ off $\\mathbf{t}_s$, then renormalize", C_TAG)
box(xa, "avg", HALF, "Average over sections", C_ENC)
box(xb, "avg", HALF, "15 section counts", C_IN)
box(M_, "cat", W - 2 * M_, "Concatenate: 768 text + 15 count features, then\nstandardize with training-split statistics", C_FEAT)
box(M_, "hgs", W - 2 * M_, "HGS over 336 logistic-regression models\nFitness: OWA of fold scores minus gap penalty\n(5-fold CV on the training split)", C_SRCH)
box(0.25, "out", W - 0.50, "Decision for T1, T1b, T2, T3 or the four T4 classes", C_OUT)
down(xb + HALF / 2, "in", "tag"); down(xa + HALF / 2, "tag", "avg")
down(xa + HALF / 2, "avg", "cat"); down(xb + HALF / 2, "avg", "cat")
down(W / 2, "cat", "hgs"); down(W / 2, "hgs", "out")
fig.canvas.draw(); rend = fig.canvas.get_renderer()
for t, (x, y0, w, h) in BOXES:                             # every label must lie inside its box
    bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
    assert bb.x0 >= x + 0.03 and bb.x1 <= x + w - 0.03 and bb.y0 >= y0 + 0.02 and bb.y1 <= y0 + h - 0.02, \
        ("text outside box", t.get_text()[:40], [round(v, 3) for v in (bb.x0, bb.x1, bb.y0, bb.y1)], (x, y0, w, h))
save(fig, "fig2_pipeline")

# ------------------------------------------------------------------ Fig. 3  pooled T3 confusion matrix
t3 = M[(M.task == "T3") & (M.strategy == "HGS")]
cm = np.array([[t3.TN.sum(), t3.FP.sum()], [t3.FN.sum(), t3.TP.sum()]])
fig, ax = plt.subplots(figsize=(2.6, 2.15))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=2000)
for i in range(2):
    for j in range(2):
        ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > 1000 else "black", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Fake"]); ax.set_yticks([0, 1])
ax.set_yticklabels(["Legitimate", "Fake"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.tight_layout(pad=0.2); save(fig, "fig4_t3_confusion")

# ------------------------------------------------------------------ Fig. 4  search strategies
fig, ax = plt.subplots(figsize=(COLW, 2.5))
off = {"HGS": -0.2, "Random": 0.0, "Grid": 0.2}
sty = {"HGS": dict(marker="o", fc="#1f78b4", ec="black", s=22),
       "Random": dict(marker="s", fc="#fdae61", ec="black", s=20),
       "Grid": dict(marker="D", fc="white", ec="#d7191c", s=22)}
lab = {"HGS": "HGS", "Random": "Random search", "Grid": "Exhaustive (seed 42)"}
for i, t in enumerate(TASKS):
    for s, o in off.items():
        g = M[(M.task == t) & (M.strategy == s)]
        if not len(g): continue
        st = sty[s]
        ax.scatter(np.full(len(g), i + o), g.f1_pos, marker=st["marker"], s=st["s"], facecolor=st["fc"],
                   edgecolor=st["ec"], lw=0.7, zorder=3, label=lab[s] if i == 0 else None)
        if s != "Grid":
            ax.plot([i + o - 0.08, i + o + 0.08], [g.f1_pos.mean()] * 2, color="black", lw=1.2, zorder=4)
ax.set_xticks(range(4)); ax.set_xticklabels(TASKS); ax.set_ylabel("Positive-class F1")
ax.set_ylim(0.945, 1.0); ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=0.8,
          handletextpad=0.2, borderaxespad=0.2)
fig.tight_layout(pad=0.2); save(fig, "fig6_search_strategies")

# ------------------------------------------------------------------ Fig. 5  embedding screen
names = {"V1": "Plain mean", "V2": "R1 renormalization", "V3": "R2, global mean", "V4": "R2, per-section mean",
         "V5": "Tag-direction projection", "V6": "ZCA whitening", "V7": "STE + cosine coherence",
         "V8": "R2 per section + coherence", "V9": "STE + L1 distances"}
base = E[E.variant == "V0"].set_index(["task", "seed"]).f1_pos
D = np.zeros((9, 4)); Wn = np.zeros((9, 4), dtype=int)
for r, v in enumerate(names):
    for c, t in enumerate(TASKS):
        x = E[(E.variant == v) & (E.task == t)].set_index(["task", "seed"]).f1_pos
        d = (x - base.loc[x.index]).values; D[r, c] = 100 * d.mean(); Wn[r, c] = (d > 0).sum()
fig, ax = plt.subplots(figsize=(COLW - 0.08, 3.3))
ax.imshow(np.clip(D, -3, 3), cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
for r in range(9):
    for c in range(4):
        ax.text(c, r, "%+.2f\n%d/3" % (D[r, c], Wn[r, c]), ha="center", va="center", fontsize=8,
                color="white" if abs(D[r, c]) > 2 else "black", linespacing=1.0)
ax.set_xticks(range(4)); ax.set_xticklabels(TASKS); ax.xaxis.tick_top()
ax.set_yticks(range(9)); ax.set_yticklabels(list(names.values()))
ax.tick_params(length=0)
fig.tight_layout(pad=0.2); save(fig, "fig7_embedding_screen")

# ------------------------------------------------------------------ Fig. 4  against the literature
fig, ax = plt.subplots(figsize=(COLW - 0.05, 2.5))
pub = {"T1": (0.9639, 0.9708), "T1b": (0.9417, 0.9633), "T3": (0.977, 0.982), "T4": (0.9608, 0.9608)}
met = {"T1": "f1_pos", "T1b": "accuracy", "T3": "f1_pos"}
for i, t in enumerate(["T1", "T1b", "T3", "T4"]):
    lo, hi = pub[t]
    ax.add_patch(plt.Rectangle((i - 0.32, lo), 0.64, max(hi - lo, 0.0012), color="#bdbdbd", zorder=1,
                               label="Published" if i == 0 else None))
    if t != "T4":
        a = HG[HG.task == t][met[t]]; b = B[(B.task == t) & (B.strategy == "HGS")][met[t]]
        ax.errorbar(i - 0.12, a.mean(), yerr=a.std(ddof=1), fmt="o", color="#1f78b4", mec="black", mew=0.6,
                    ms=5, capsize=2, lw=0.9, zorder=3, label="STE (ours)" if i == 0 else None)
        ax.errorbar(i + 0.12, b.mean(), yerr=b.std(ddof=1), fmt="^", color="#fdae61", mec="black", mew=0.6,
                    ms=5.5, capsize=2, lw=0.9, zorder=3, label="Projection (ours)" if i == 0 else None)
    else:
        same = P[(P.split == "random")].set_index("variant").macro_f1
        strat = P[(P.split == "stratified") & (P.variant == "V0")].macro_f1
        ax.plot(i - 0.12, same["V0"], "o", color="#1f78b4", mec="black", mew=0.6, ms=5, zorder=3)
        ax.plot(i + 0.12, same["V5"], "^", color="#fdae61", mec="black", mew=0.6, ms=5.5, zorder=3)
        ax.errorbar(i + 0.36, strat.mean(), yerr=strat.std(ddof=1), fmt="o", mfc="white", mec="#1f78b4",
                    color="#1f78b4", mew=1.0, ms=5, capsize=2, lw=0.9, zorder=3, label="STE, stratified")
ax.set_xticks(range(4)); ax.set_xticklabels(["T1\nF1", "T1b\naccuracy", "T3\nF1", "T4\nmacro F1"])
ax.set_xlim(-0.55, 3.62); ax.set_ylim(0.935, 0.995); ax.set_ylabel("Score")
ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.3)
fig.tight_layout(pad=0.2); save(fig, "fig5_literature")

# ------------------------------------------------------------------ Fig. 7  projection against STE, paired by split
fig, axs = plt.subplots(2, 2, figsize=(COLW - 0.05, 2.95)); axs = axs.ravel()
SEEDC = {42: "#08519c", 7: "#6baed6", 123: "#fd8d3c"}; SEEDM = {42: "o", 7: "s", 123: "D"}
for ax, t in zip(axs, TASKS):
    s = HG[HG.task == t].set_index("seed").f1_pos; p = B[(B.task == t) & (B.strategy == "HGS")].set_index("seed").f1_pos
    for sd in [42, 7, 123]:
        ax.plot([0, 1], [s[sd], p[sd]], "-", color=SEEDC[sd], lw=1.0, marker=SEEDM[sd], ms=3.5, mec="black", mew=0.4,
                label=("seed %d" % sd) if t == "T1" else None)
    ax.set_title("%s  %+.2f" % (t, 100 * (p - s).mean()), fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["STE", "Proj."]); ax.set_xlim(-0.35, 1.35)
    lo, hi = min(s.min(), p.min()), max(s.max(), p.max()); m = (hi - lo) * 0.25 + 0.001
    ax.set_ylim(lo - m, hi + m); ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
axs[0].set_ylabel("F1"); axs[2].set_ylabel("F1")
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.9)
fig.tight_layout(pad=0.2, w_pad=0.8, h_pad=0.6); save(fig, "fig8_projection_pairs")

# ------------------------------------------------------------------ Fig. 8  four-class confusion matrices
strat = sum(np.array(json.loads(c)) for c in P[(P.split == "stratified") & (P.variant == "V0")].confusion)
same = np.array(json.loads(P[(P.split == "random") & (P.variant == "V0")].confusion.iloc[0]))
fig, axs = plt.subplots(1, 2, figsize=(COLW - 0.05, 2.15))
codes = ["0", "1", "10", "11"]
for ax, cm, title in zip(axs, [strat, same], ["(a) Stratified, 3 splits", "(b) Published class counts"]):
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "%d" % cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(codes); ax.set_yticks(range(4)); ax.set_yticklabels(codes)
    ax.set_title(title, fontsize=8); ax.set_xlabel("Predicted code"); ax.tick_params(length=0)
axs[0].set_ylabel("True code")
fig.tight_layout(pad=0.2, w_pad=0.6); save(fig, "fig9_four_class_confusion")

# ------------------------------------------------------------------ Fig. 3  OWA weights
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
fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.62)); lbl = ["weak", "mid", "strong"]
for k, o in enumerate([0.10, 0.31, 0.50, 0.90]):
    ax[0].plot(range(3), owa_exp(3, o), marker="os^d"[k], ms=4, lw=1.2, label="%.2f" % o,
               color=["#08306b", "#4292c6", "#fd8d3c", "#a63603"][k], mec="black", mew=0.4)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lbl, fontsize=8); ax[0].set_ylabel("weight"); ax[0].set_ylim(0, 0.95)
ax[0].legend(fontsize=8, frameon=False, handlelength=1.4); ax[0].set_title("(a) By orness", fontsize=8)
we, wm = owa_exp(3, 0.30), owa_maxent(3, 0.30)[::-1]; x = np.arange(3)
ax[1].bar(x - 0.18, we, 0.36, label="Exp., (4)", color="#1f78b4", edgecolor="k", lw=0.4)
ax[1].bar(x + 0.18, wm, 0.36, label="Max. ent.", color="#fdae61", edgecolor="k", lw=0.4, hatch="//")
ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=8); ax[1].set_ylim(0, 0.8)
ax[1].legend(fontsize=8, frameon=False); ax[1].set_title("(b) Orness 0.30", fontsize=8)
for a in ax: a.tick_params(labelsize=8); a.grid(axis="y", lw=0.3, alpha=0.4)
fig.tight_layout(pad=0.25); save(fig, "fig3_owa_weights")
print("  exponential weights at orness 0.30:", np.round(we, 4), "| maximum entropy:", np.round(wm, 4))

