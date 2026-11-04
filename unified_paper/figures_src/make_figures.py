"""Regenerates the new figures for the unified paper from raw run artifacts.

Inputs (read-only, all from the public Hugging Face repos cloned locally):
  VG_CODE  = Dhruv1000/vulnguard-code-and-report   (final_scientific_audit_20260927/*, campaign_D5/*)
  TG       = Dhruv1000/tracegrpo                   (results/phase10/final_test_20261012/test_final_metrics.json)
Usage: python make_figures.py <VG_CODE_DIR> <TG_DIR> <OUT_DIR>
"""
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VG, TG, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)

# Validated reference palette, light mode, first three slots (all-pairs safe).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
})

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Fig: ROC-space of every system
tg = json.load(open(TG / "results/phase10/final_test_20261012/test_final_metrics.json"))["systems"]
def tgpt(k):
    pc = tg[k]["per_class_accuracy"]
    return 1 - pc["SAFE"]["accuracy"], pc["VULNERABLE"]["accuracy"]

sup = {r["run_id"]: r for r in csv.DictReader(open(VG / "results/final_scientific_audit_20260927/supporting_experiments.csv"))}
def vgpt(run):
    m = json.loads(sup[run]["metrics"])
    return m["vulnerable_prediction_on_patched"], m["vulnerable_prediction_on_vulnerable"]
prim = list(csv.DictReader(open(VG / "results/final_scientific_audit_20260927/primary_seed_results.csv")))
fresh = {}
for r in prim:
    key = (r["condition"], int(r["seed"]))
    fresh[key] = (float(r["vulnerable_prediction_rate_patched"]) if "vulnerable_prediction_rate_patched" in r else None,
                  float(r["vulnerable_prediction_rate_vulnerable"]) if "vulnerable_prediction_rate_vulnerable" in r else None)

fresh = {(r["condition"], int(r["seed"])): (float(r["vulnerable_prediction_rate_patched"]),
                                            float(r["vulnerable_prediction_rate_vulnerable"])) for r in prim}

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.plot([0, 1], [0, 1], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.text(0.18, 0.27, "no discrimination", rotation=43, color=INK2, fontsize=7, ha="center", va="center")
ax.fill_between([0, 1], [0, 1], [1, 1], color="#f3f2ee", zorder=0)
ax.text(0.03, 0.80, "above the line:\ndiscriminates", fontsize=7, color=INK2, va="top")

def pt(xy, color, marker, **kw):
    ax.scatter(*xy, s=kw.pop("s", 34), color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3, **kw)

# Program A (TraceGRPO), chronological TEST, frozen vote rule
pt(tgpt("sft500"), BLUE, "o"); pt(tgpt("grpo324"), BLUE, "o"); pt(tgpt("e4b_step200"), BLUE, "o")
pt(tgpt("classical"), AQUA, "D")
# Program B (VulnGuard), PrimeVul time-test, per-sample rates
for (c, s), xy in fresh.items():
    pt(xy, ORANGE, "o" if c == "gold" else "s")
pt(vgpt("f260926184024-hist-permuted0-test"), ORANGE, "s")
pt(vgpt("f260926184024-hist-binary0-test"), ORANGE, "^")
pt(vgpt("f260926184024-hist-flipped0-test"), ORANGE, "P")
pt(vgpt("f260926184024-hist-base-test"), ORANGE, "v")

def lab(xy, text, to, ha="left", va="center"):
    ax.annotate(text, xy, xytext=to, fontsize=6.5, color=INK, ha=ha, va=va,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=3))
lab(tgpt("sft500"), "SFT-500", (0.58, 0.30))
lab(tgpt("classical"), "TF-IDF (classical)", (0.68, 0.40))
lab(vgpt("f260926184024-hist-binary0-test"), "BINARY", (0.40, 0.12))
lab(tgpt("e4b_step200"), "E4b-200 (A) and\nPERMUTED s0 (B)", (0.80, 0.52))
lab(vgpt("f260926184024-hist-base-test"), "untrained base", (0.88, 0.68))
lab(tgpt("grpo324"), "GRPO-324", (0.90, 0.76))
lab(fresh[("permuted", 3)], "PERMUTED (3 fresh)", (0.93, 0.84), ha="left")
lab(vgpt("f260926184024-hist-flipped0-test"), "FLIPPED", (0.62, 0.95), ha="right")
lab((0, 0.0009), "GOLD (3 fresh, 2 hist.)", (0.12, 0.07))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
ax.set_xlabel("P(VULNERABLE verdict | patched / SAFE input)")
ax.set_ylabel("P(VULNERABLE verdict | vulnerable input)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", ms=7, label="Program A (TraceGRPO), neural"),
     Line2D([], [], marker="D", color="none", markerfacecolor=AQUA, markeredgecolor="white", ms=6, label="Program A, classical baseline"),
     Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", ms=7, label="Program B (VulnGuard), gold / base"),
     Line2D([], [], marker="s", color="none", markerfacecolor=ORANGE, markeredgecolor="white", ms=6, label="Program B, placebo / other reward arms")]
ax.legend(handles=h, loc="upper center", fontsize=6.3, bbox_to_anchor=(0.5, -0.17), ncol=2, handletextpad=0.3, columnspacing=0.8)
save(fig, "fig_roc_space")

# ---------------------------------------------------------------- Fig: VulnGuard training verdict-rate trajectories
traj = list(csv.DictReader(open(VG / "results/final_scientific_audit_20260927/training_trajectories.csv")))
byrun = {}
for r in traj:
    if r["vulnerable_prediction_rate_overall_valid"] == "":
        continue  # step 40 has no logged rollout file; left out, not imputed
    byrun.setdefault(r["run_id"], []).append((int(r["step"]), float(r["vulnerable_prediction_rate_overall_valid"])))
def s1(name):
    rows = [json.loads(l) for l in open(VG / f"results/campaign_D5/{name}/metrics.jsonl")]
    return [r for r in rows if "diag/verdict_prior_vuln" in r and "step" in r]

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True)
ax = axs[0]
ls = {2: "-", 3: (0, (5, 2)), 4: (0, (1, 1.5))}
for rid, pts in sorted(byrun.items()):
    arm = "gold" if "-gold-" in rid else "permuted"; seed = int(rid[-1])
    pts.sort(); ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE if arm == "gold" else ORANGE, ls=ls[seed], lw=1.4)
ax.axhspan(0.45, 0.55, color="#e9e8e3", zorder=0)
ax.set_title("(a) Final cohort: 3 + 3 matched seeds, plain GRPO", fontsize=8, loc="left")
ax.set_xlabel("optimizer step"); ax.set_ylabel("VULNERABLE verdict rate (training rollouts)")
ax.set_xlim(0, 46); ax.set_ylim(-0.02, 1.02)
ax.text(39.5, 0.50, "pre-reg.\nband", fontsize=6, va="center", color=INK2)
ax.text(30, 0.07, "GOLD (s2, s3, s4)", color=BLUE, fontsize=7); ax.text(2, 0.93, "PERMUTED (s2, s3, s4)", color=ORANGE, fontsize=7)
ax = axs[1]
styles = {"s1a-h4-gold-clamp": (BLUE, "-", "GOLD + verdict clamp (H4)"), "s1b-h4-perm-clamp": (ORANGE, "-", "PERMUTED + clamp"),
          "s1c-h5-gold-pair": (BLUE, (0, (5, 2)), "GOLD + pair-symmetric (H5)"), "s1d-h5-perm-pair": (ORANGE, (0, (5, 2)), "PERMUTED + pair-symmetric")}
for name, (c, l, t) in styles.items():
    rows = s1(name)
    ax.plot([r["step"] for r in rows], [r["diag/verdict_prior_vuln"] for r in rows], color=c, ls=l, lw=1.4, label=t)
ax.axhspan(0.45, 0.55, color="#e9e8e3", zorder=0)
ax.set_title("(b) Campaign D5: two purpose-built rescues", fontsize=8, loc="left")
ax.set_xlabel("optimizer step"); ax.set_xlim(0, 62)
ax.legend(fontsize=6.3, loc="center right", bbox_to_anchor=(1.0, 0.33), handlelength=2.6)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_vg_trajectories")

# ---------------------------------------------------------------- Fig: H4 mechanism (S1a solid, S1b dashed)
a, b = s1("s1a-h4-gold-clamp"), s1("s1b-h4-perm-clamp")
fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9))
def two(ax, key, ylabel, title, ylim=None):
    ax.plot([r["step"] for r in a], [r[key] for r in a], color=BLUE, lw=1.4)
    ax.plot([r["step"] for r in b], [r[key] for r in b], color=ORANGE, lw=1.4, ls=(0, (5, 2)))
    ax.set_title(title, fontsize=7.5, loc="left"); ax.set_xlabel("step"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
two(axs[0], "constraint/rate1_vuln", "VULNERABLE rate", "(a) verdict rate", (-0.02, 0.8))
axs[0].axhspan(0.45, 0.55, color="#e9e8e3", zorder=0)
two(axs[1], "constraint/lambda2", r"$\lambda_2$ (SAFE penalty)", r"(b) dual variable $\lambda_2$")
two(axs[2], "entropy", "policy entropy", "(c) entropy")
two(axs[3], "rewards/content/mean", "mean content reward", "(d) content reward")
axs[3].axhline(0.5, color=INK2, lw=0.8, ls=(0, (2, 2))); axs[3].text(60, 0.455, "0.5 = value of\nalways-SAFE", fontsize=6, color=INK2, ha="right", va="top")
from matplotlib.lines import Line2D as _L
fig.legend(handles=[_L([],[],color=BLUE,lw=1.4,label="GOLD + clamp (S1a)"),_L([],[],color=ORANGE,lw=1.4,ls=(0,(5,2)),label="PERMUTED + clamp (S1b)")], loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5,-0.12))
fig.tight_layout(w_pad=0.8)
save(fig, "fig_h4_mechanism")

# ---------------------------------------------------------------- Fig: strict P-C@1 on the held-out TEST pairs
pc = {}
for rid, r in sup.items():
    if rid.startswith("f260926184024") and rid.endswith("-test"):
        pc[rid] = json.loads(r["metrics"])["pair_correct"]
rows = [("untrained base", [pc["f260926184024-hist-base-test"]], AQUA, "v"),
        ("GOLD fresh (s2-4)", [pc[f"f260926184024-gold-s{i}-test"] for i in (2, 3, 4)], BLUE, "o"),
        ("GOLD historical (s0 D2-std, s1 A1)", [pc["f260926184024-hist-gold0-test"], pc["f260926184024-hist-gold1-test"]], BLUE, "o"),
        ("PERMUTED fresh (s2-4)", [pc[f"f260926184024-permuted-s{i}-test"] for i in (2, 3, 4)], ORANGE, "s"),
        ("PERMUTED historical (s0)", [pc["f260926184024-hist-permuted0-test"]], ORANGE, "s"),
        ("BINARY historical (s0)", [pc["f260926184024-hist-binary0-test"]], ORANGE, "^"),
        ("FLIPPED historical (s0)", [pc["f260926184024-hist-flipped0-test"]], ORANGE, "P")]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for i, (name, vals, c, m) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.scatter([100 * v for v in vals], [y] * len(vals), color=c, marker=m, s=30, edgecolor="white", linewidth=0.7, zorder=3)
    ax.plot([100 * min(vals), 100 * max(vals)], [y, y], color=c, lw=1.0, alpha=0.5, zorder=2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.8)
ax.set_xlabel("strict localized P-C@1 on held-out TEST pairs (%)"); ax.set_xlim(-0.3, 6.2)
ax.grid(axis="y", visible=False)
save(fig, "fig_pc1_endpoints")
print("ok", {k: round(v, 4) for k, v in pc.items()})

# ---------------------------------------------------------------- Fig: closed-form equilibrium q*(kappa), Program A
cf = json.load(open(TG / "results/phase6/review22/reward_counterfactuals.json"))
eq = cf["reward_implied_prior_equilibrium"]["scale_5pct_steps201plus"]
import numpy as np
def qstar(kappa, dv, ds): return kappa * dv / (kappa * dv + ds)
cur, sep = eq["current"], eq["evidence_separated"]
dv_c, ds_c = cur["mean_correct_minus_wrong_gold_VULNERABLE"], cur["mean_correct_minus_wrong_gold_SAFE"]
dv_s, ds_s = sep["mean_correct_minus_wrong_gold_VULNERABLE"], sep["mean_correct_minus_wrong_gold_SAFE"]
# sanity: closed form must reproduce the stored values
assert abs(qstar(1.5, dv_c, ds_c) - cur["closed_form_q_star_label_weight_1.5"]) < 1e-9
assert abs(qstar(1.0, dv_s, ds_s) - sep["closed_form_q_star_label_weight_1.0"]) < 1e-9
prod = [r["fraction_predicted_VULNERABLE"] for r, st in zip(cf["observed_sampling_rates"]["scale_5pct_by_50_steps"],
        [sum(r["adapter_steps"]) / 2 for r in cf["observed_sampling_rates"]["scale_5pct_by_50_steps"]]) if st >= 200]
e4 = [json.loads(l) for l in open(TG / "results/phase6/review26/e4b_balanced_reward/step_metrics.jsonl")]
e4b = [sum(r["sampled_train_vulnerable_prediction_rate"] for r in e4 if lo <= r["optimizer_step"] < lo + 50) /
       max(1, sum(1 for r in e4 if lo <= r["optimizer_step"] < lo + 50)) for lo in range(1, 201, 50)]

k = np.linspace(0.6, 2.0, 200)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (2, 2)))
ax.plot(k, qstar(k, dv_c, ds_c), color=BLUE, lw=1.5, label="production reward")
ax.plot(k, qstar(k, dv_s, ds_s), color=ORANGE, lw=1.5, ls=(0, (5, 2)), label="evidence-separated reward")
ax.scatter([1.5], [cur["closed_form_q_star_label_weight_1.5"]], color=BLUE, s=34, zorder=4, edgecolor="white", linewidth=0.8)
ax.scatter([1.0], [sep["closed_form_q_star_label_weight_1.0"]], color=ORANGE, marker="s", s=34, zorder=4, edgecolor="white", linewidth=0.8)
ax.vlines(1.5, min(prod), max(prod), color=BLUE, lw=3.5, alpha=0.35)
ax.vlines(1.0, min(e4b), max(e4b), color=ORANGE, lw=3.5, alpha=0.35)
ax.annotate("production, "+r"$\kappa$=1.5"+":\npredicted 0.532;\nobserved %.3f-%.3f" % (min(prod), max(prod)), (1.5, 0.52), (1.62, 0.405), fontsize=6.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
ax.annotate("E4b, "+r"$\kappa$=1"+":\npredicted 0.498;\nobserved %.3f-%.3f" % (min(e4b), max(e4b)), (1.0, 0.485), (0.62, 0.39), fontsize=6.3,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
ax.set_xlabel(r"vulnerable-class label weight $\kappa$"); ax.set_ylabel(r"equilibrium VULNERABLE vote rate $q^*$")
ax.set_ylim(0.36, 0.66); ax.legend(fontsize=6.3, loc="upper left")
save(fig, "fig_equilibrium")
print("prod", [round(x, 4) for x in prod], "e4b", [round(x, 4) for x in e4b])
