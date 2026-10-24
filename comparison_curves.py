"""Fig. 6 - training return (sparse) and policy entropy for the nine 8x8 comparison runs.

Reads runs/cmp_{control,shaped,fullobs}_8x8_seed{1,2,3}/metrics.csv and writes
fig6_comparison.png next to this script.

    python comparison_curves.py
"""
import csv, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.abspath(__file__))
B=os.path.join(ROOT,"runs")
C={"control":"#2a78d6","shaped":"#eb6834","fullobs":"#1baf7a"}
NAME={"control":"G1 control","shaped":"G2 shaped","fullobs":"G3 full view"}
LS={1:"-",2:"--",3:":"}
INK2="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.size":8.5,"axes.edgecolor":INK2,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
def series(g,s,col):
    m=list(csv.DictReader(open(f"{B}/cmp_{g}_8x8_seed{s}/metrics.csv")))
    x=np.array([int(r["global_step"]) for r in m])/1e3
    y=np.array([float(r[col]) if r[col] else np.nan for r in m])
    if col=="episodic_return_mean": y=np.nan_to_num(y,nan=0.0)
    k=9; y=np.convolve(np.pad(y,(k//2,k//2),mode="edge"),np.ones(k)/k,mode="valid")
    return x,y
fig,axs=plt.subplots(1,2,figsize=(7.0,2.5))
for ax,col,title,yl in [(axs[0],"episodic_return_mean","Training return (sparse, smoothed)","Mean episodic return"),
                        (axs[1],"entropy","Policy entropy","Entropy (nats)")]:
    for g in ["control","fullobs","shaped"]:
        for s in (1,2,3):
            x,y=series(g,s,col); ax.plot(x,y,color=C[g],ls=LS[s],lw=1.4,label=NAME[g] if s==1 else None)
    ax.set_title(title,fontsize=8.5,loc="left"); ax.set_xlabel("Environment steps (thousands)"); ax.set_ylabel(yl)
axs[1].axhline(np.log(7),color=INK2,lw=0.7,ls="--"); axs[1].text(1000,np.log(7)+0.05,"ln 7",ha="right",fontsize=7,color=INK2)
h,l=axs[0].get_legend_handles_labels(); o=[l.index(n) for n in ("G1 control","G2 shaped","G3 full view")]
axs[0].legend([h[i] for i in o],[l[i] for i in o],fontsize=7.2,loc="center right")
fig.tight_layout(); out=os.path.join(ROOT,"fig6_comparison.png"); fig.savefig(out,dpi=200); print("wrote",out)
