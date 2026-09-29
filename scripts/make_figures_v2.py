"""Figures added in version 1.2.0 of the paper (Fig. 4 with 95% bootstrap intervals, Fig. 6 projection effect).

The plotted values are copied from results/stats_intervals.csv, results/stats_report.txt,
results/confirm_summary.txt, results/groupsplit_report.txt and the EMSCAD report of notebook 14.
Usage:  python scripts/make_figures_v2.py   (writes PNG and JPEG files in the working directory)
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.family": "serif","font.serif": ["Times New Roman","Liberation Serif","Nimbus Roman","DejaVu Serif"],
    "font.size": 8,"axes.titlesize": 8,"axes.labelsize": 8,"xtick.labelsize": 8,"ytick.labelsize": 8,"legend.fontsize": 8,"mathtext.fontset":"stix"})
COLW=3.35
def save(fig,name):
    for t in fig.findobj(matplotlib.text.Text):
        assert not (t.get_text().strip() and t.get_visible() and t.get_fontsize()<8), t.get_text()
    fig.savefig(name+".png",dpi=600,bbox_inches="tight",pad_inches=0.03,facecolor="white")
    im=Image.open(name+".png").convert("RGB"); w=im.size[0]/600; assert w<=COLW+0.02,(name,w)
    im.save(name+".jpeg",quality=95,dpi=(600,600)); print(name,im.size,round(w,2),round(im.size[1]/600,2))

# ---- Fig. 4: against the literature, 95% bootstrap intervals
fig, ax = plt.subplots(figsize=(COLW-0.05, 2.45))
pub={"T1":(0.9639,0.9708),"T1b":(0.9417,0.9633),"T3":(0.977,0.982),"T4":(0.9608,0.9608)}
ste={"T1":(0.9705,0.9604,0.9797),"T1b":(0.9787,0.9694,0.9870),"T3":(0.9812,0.9763,0.9858),"T4":(0.9616,0.9456,0.9762)}
prj={"T1":(0.9685,0.9577,0.9785),"T1b":(0.9778,0.9685,0.9861),"T3":(0.9849,0.9807,0.9889),"T4":(0.9601,0.9436,0.9753)}
for i,t in enumerate(["T1","T1b","T3","T4"]):
    lo,hi=pub[t]
    ax.add_patch(plt.Rectangle((i-0.32,lo),0.64,max(hi-lo,0.0012),color="#bdbdbd",zorder=1,label="Published" if i==0 else None))
    for o,d,fmt,col,ms,lab in [(-0.12,ste,"o","#1f78b4",5,"STE (ours)"),(0.12,prj,"^","#fdae61",5.5,"Projection (ours)")]:
        m,a,b=d[t]
        ax.errorbar(i+o,m,yerr=[[m-a],[b-m]],fmt=fmt,color=col,mec="black",mew=0.6,ms=ms,capsize=2,lw=0.9,zorder=3,label=lab if i==0 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(["T1\nF1","T1b\naccuracy","T3\nF1","T4\nmacro F1"])
ax.set_xlim(-0.55,3.55); ax.set_ylim(0.935,0.995); ax.set_ylabel("Score")
ax.grid(axis="y",lw=0.3,alpha=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=3,frameon=False,columnspacing=0.8,handletextpad=0.3)
fig.tight_layout(pad=0.2); save(fig,"fig_literature_ci")

# ---- Fig. 6: every paired measurement of the projection effect (points of F1)
rows=[("LinkedIn T1, random splits",[0.02,-0.63,-0.02],None),
      ("LinkedIn T1b, random splits",[-0.30,0.28,-0.29],None),
      ("LinkedIn T2, random splits",[0.14,-0.14,-0.41],None),
      ("LinkedIn T3, random splits",[-0.09,0.47,0.74],None),
      ("LinkedIn T3, pre-registered",[0.09,0.47,0.00,0.92,-0.01,0.46,0.18,0.00,0.19,0.37],(0.11,0.45)),
      ("LinkedIn T1, template groups",[-0.41,0.20,0.37],None),
      ("LinkedIn T1b, template groups",[0.87,0.29,-0.02],None),
      ("LinkedIn T3, template groups",[0.20,0.26,-0.18],None),
      ("EMSCAD, deduplicated",[2.43,-1.38,-1.63],None)]
fig, ax = plt.subplots(figsize=(COLW-0.05, 2.75))
n=len(rows)
for k,(lab,v,ci) in enumerate(rows):
    y=n-1-k; v=np.array(v)
    if k in (4,): ax.axhspan(y-0.5,y+0.5,color="#fff3e0",zorder=0)
    jit=np.linspace(-0.16,0.16,len(v)) if len(v)>3 else np.array([-0.12,0,0.12])
    ax.scatter(v,y+jit,s=9,facecolor="#9ecae1",edgecolor="#08519c",lw=0.4,zorder=2)
    if ci: ax.plot(ci,[y,y],color="black",lw=1.2,zorder=3)
    ax.plot(v.mean(),y,"D",color="#d95f02",mec="black",mew=0.5,ms=4.5,zorder=4)
ax.axvline(0,color="#555555",lw=0.7,ls="--",zorder=1)
for yb in (4.5,3.5,0.5): ax.axhline(yb,color="#cccccc",lw=0.5)
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlabel("F1 difference, projection minus STE (points)", x=0.3); ax.set_xlim(-2.0,2.8)
ax.scatter([],[],s=9,facecolor="#9ecae1",edgecolor="#08519c",lw=0.4,label="One split")
ax.plot([],[],"D",color="#d95f02",mec="black",mew=0.5,ms=4.5,label="Mean",ls="none")
ax.plot([],[],color="black",lw=1.2,label="95% CI")
ax.legend(loc="lower center",bbox_to_anchor=(0.35,1.0),ncol=3,frameon=False,columnspacing=0.8,handletextpad=0.3)
ax.tick_params(axis="y",length=0)
fig.tight_layout(pad=0.2); save(fig,"fig_projection_forest")
