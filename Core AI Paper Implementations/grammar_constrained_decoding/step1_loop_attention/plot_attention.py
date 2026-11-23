"""Task B analysis: per-sequence image-attention curves, onset-aligned average, statistics.
Layer groups: ALL (0-35), POST_DS (layers >= 3, i.e. after DeepStack injection into layers 0-2),
LATE (layers 24-35). Main curve = POST_DS mean over layers and heads.
Writes plots/*.png and attn_summary.csv, attn_onset_stats.json.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path("/mnt/work/step1_loops_attention")
PL = WORK / "plots"; PL.mkdir(exist_ok=True)
sel = pd.read_csv(WORK / "attn_selection.csv")
env = json.load(open(WORK / "attn_env.json"))
NL = env["n_layers"]; DS = len(env["deepstack_visual_indexes"])
GROUPS = {"ALL": list(range(NL)), "POST_DS": list(range(DS, NL)), "LATE": list(range(NL - 12, NL))}
W, SM = 200, 25
CATS = ["sink", "image", "prompt", "gen"]


def smooth(x, k=SM):
    if len(x) < k: return x
    return np.convolve(x, np.ones(k) / k, mode="same")


data = {}
for _, r in sel.iterrows():
    f = WORK / "attn" / f"{r.req_id}.npz"
    if f.exists():
        z = np.load(f); data[r.req_id] = (r, z["frac"].astype(np.float32), float(z["argmax_agree"]))

rows = []
for rid, (r, fr, agree) in data.items():
    rec = dict(req_id=rid, role=r.role, condition=r.condition, n_gen=fr.shape[1], argmax_agree=agree,
               onset=r.loop_onset_token if r.role == "loop" else np.nan)
    for g, ls in GROUPS.items():
        img = fr[ls, :, 1].mean(0)
        rec[f"img_mean_{g}"] = img.mean()
        rec[f"img_first200_{g}"] = img[:200].mean()
        rec[f"img_last200_{g}"] = img[-200:].mean()
        if r.role == "loop":
            o = int(r.loop_onset_token)
            rec[f"img_pre_{g}"] = img[max(0, o - W):o].mean() if o > 0 else np.nan
            rec[f"img_post_{g}"] = img[o:o + W].mean()
            rec[f"gen_pre_{g}"] = fr[ls, max(0, o - W):o, 3].mean() if o > 0 else np.nan
            rec[f"gen_post_{g}"] = fr[ls, o:o + W, 3].mean()
    rows.append(rec)
summ = pd.DataFrame(rows)
summ.to_csv(WORK / "attn_summary.csv", index=False)

# paired control: same absolute token windows as its loop partner (controls the length confound)
loops = sel[sel.role == "loop"].reset_index(drop=True)
ctrls = sel[sel.role == "control"].reset_index(drop=True)
pairs = []
for i, lr in loops.iterrows():
    if i >= len(ctrls) or lr.req_id not in data or ctrls.req_id[i] not in data: continue
    o = int(lr.loop_onset_token)
    fl = data[lr.req_id][1][GROUPS["POST_DS"], :, 1].mean(0)
    fc = data[ctrls.req_id[i]][1][GROUPS["POST_DS"], :, 1].mean(0)
    def win(x, a, b):
        a, b = max(0, a), min(len(x), b)
        return x[a:b].mean() if b > a else np.nan
    pairs.append(dict(loop=lr.req_id, control=ctrls.req_id[i], onset=o,
                      loop_pre=win(fl, o - W, o), loop_post=win(fl, o, o + W),
                      ctrl_pre=win(fc, o - W, o), ctrl_post=win(fc, o, o + W)))
pairs = pd.DataFrame(pairs); pairs.to_csv(WORK / "attn_paired_windows.csv", index=False)

# --- per-sequence plots
for rid, (r, fr, agree) in data.items():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for g, c in [("POST_DS", "C0"), ("LATE", "C2"), ("ALL", "0.6")]:
        ax.plot(smooth(fr[GROUPS[g], :, 1].mean(0)), color=c, lw=1.2, label=f"image frac ({g})")
    ax.plot(smooth(fr[GROUPS["POST_DS"], :, 3].mean(0)), color="C3", lw=1, ls="--", label="gen-text frac (POST_DS)")
    if r.role == "loop":
        ax.axvline(r.loop_run_start_token, color="k", ls=":", lw=1, label="run start")
        ax.axvline(r.loop_onset_token, color="k", lw=1.5, label="loop onset (1st repeat)")
    ax.set_xlabel("generated token index"); ax.set_ylabel("attention fraction (head-avg)")
    ax.set_title(f"{rid} [{r.role}, {r.condition}] finish={r.finish_reason} tokens={r.completion_tokens}"
                 + (f" run={r.longest_run}" if r.role == "loop" else ""), fontsize=9)
    ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper right", ncol=2); fig.tight_layout()
    fig.savefig(PL / f"seq_{r.role}_{rid}.png", dpi=110); plt.close(fig)

# --- onset-aligned average
rel = np.arange(-W, W + 1)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
stats = {}
for g, ax in zip(["POST_DS", "LATE"], axs):
    M = []; MC = []
    for i, lr in loops.iterrows():
        if lr.req_id not in data: continue
        o = int(lr.loop_onset_token); img = data[lr.req_id][1][GROUPS[g], :, 1].mean(0)
        row = np.full(len(rel), np.nan); idx = o + rel; ok = (idx >= 0) & (idx < len(img))
        row[ok] = img[idx[ok]]; M.append(row)
        if i < len(ctrls) and ctrls.req_id[i] in data:
            ci = data[ctrls.req_id[i]][1][GROUPS[g], :, 1].mean(0)
            rc = np.full(len(rel), np.nan); okc = (idx >= 0) & (idx < len(ci)); rc[okc] = ci[idx[okc]]; MC.append(rc)
    if not M: continue
    M = np.array(M); mu = np.nanmean(M, 0); n = np.sum(~np.isnan(M), 0)
    se = np.nanstd(M, 0) / np.sqrt(np.maximum(n, 1))
    ax.plot(rel, smooth(mu, 9), label=f"looped (n={len(M)})"); ax.fill_between(rel, mu - se, mu + se, alpha=.2)
    if MC:
        MC = np.array(MC); ax.plot(rel, smooth(np.nanmean(MC, 0), 9), color="C1",
                                    label=f"controls at same abs. tokens (n={len(MC)})")
    ax.axvline(0, color="k"); ax.set_xlabel("tokens relative to loop onset"); ax.set_ylabel("image attention frac")
    ax.set_title(f"onset-aligned, layers {g}"); ax.legend(fontsize=8)
    stats[g] = {f"loop_{a}_{b}": float(np.nanmean(M[:, (rel >= a) & (rel < b)])) for a, b in
                [(-200, -100), (-100, 0), (0, 100), (100, 200)]}
fig.tight_layout(); fig.savefig(PL / "onset_aligned_image_attention.png", dpi=120); plt.close(fig)

# per-layer image fraction pre vs post onset
pre_l, post_l = [], []
for _, lr in loops.iterrows():
    if lr.req_id not in data: continue
    o = int(lr.loop_onset_token); fr = data[lr.req_id][1]
    if o >= 20: pre_l.append(fr[:, max(0, o - W):o, 1].mean(1))
    post_l.append(fr[:, o:o + W, 1].mean(1))
if post_l:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if pre_l: ax.plot(np.mean(pre_l, 0), "o-", ms=3, label="200 tok before onset")
    ax.plot(np.mean(post_l, 0), "o-", ms=3, label="200 tok after onset")
    ax.axvspan(-0.5, DS - 0.5, color="0.9", label="DeepStack-injected layers")
    ax.set_xlabel("layer"); ax.set_ylabel("image attention frac"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PL / "per_layer_image_attention_pre_post.png", dpi=120); plt.close(fig)

stats["n_loop_seqs"] = int(len([1 for r in loops.req_id if r in data]))
stats["n_ctrl_seqs"] = int(len([1 for r in ctrls.req_id if r in data]))
if len(pairs):
    p = pairs.dropna()
    stats["paired_POST_DS"] = dict(n=len(p), loop_pre=p.loop_pre.mean(), loop_post=p.loop_post.mean(),
                                   ctrl_pre=p.ctrl_pre.mean(), ctrl_post=p.ctrl_post.mean(),
                                   n_loops_post_lt_pre=int((p.loop_post < p.loop_pre).sum()),
                                   n_loop_pre_lt_ctrl_pre=int((p.loop_pre < p.ctrl_pre).sum()))
    try:
        from scipy.stats import wilcoxon
        stats["paired_POST_DS"]["wilcoxon_loop_post_vs_pre_p"] = float(wilcoxon(p.loop_post, p.loop_pre).pvalue)
        stats["paired_POST_DS"]["wilcoxon_loop_pre_vs_ctrl_pre_p"] = float(wilcoxon(p.loop_pre, p.ctrl_pre).pvalue)
    except Exception as ex:
        stats["wilcoxon_error"] = str(ex)
json.dump(stats, open(WORK / "attn_onset_stats.json", "w"), indent=1, default=float)
print(json.dumps(stats, indent=1, default=float))
print(summ.groupby("role")[[c for c in summ.columns if c.startswith("img_")]].mean().T.to_string())
