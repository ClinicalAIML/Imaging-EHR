#!/usr/bin/env python3
"""
Reproducible simulation study for Section 4.7 of
  "Fusion of medical imaging and electronic health records for disease
   classification using deep learning and statistical methods: principles,
   pitfalls, and future directions"  (submitted to Statistical Methods in
   Medical Research)

The script regenerates, from synthetic data only,
  * Table 3  (simulation results, all panels)      -> table_simulation.tex
  * Figure 4 (discrimination gain and calibration) -> figure_simulation.pdf/.png
                                                      and figure_simulation_pgfplots.tex
  * all replicate-level results                    -> simulation_results_long.csv
  * summary statistics with Monte Carlo SEs        -> simulation_results_summary.csv

All tabulated quantities are reported to three decimal places.
Every random draw comes from numpy Generators with fixed seeds (one seed per
experiment), so a rerun gives identical numbers.

Experiments: A (cross-modal covariance, seven estimators, closed-form population AUCs),
              A' (all logistic penalties tuned by CV of the log-likelihood), B (label error), C (selection).

Usage:  python3 simulation_section4_7.py            (full run, ~2.5 min on one core)
        python3 simulation_section4_7.py --quick    (10 replicates, for a smoke test)

Software used for the manuscript: Python 3.12, numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0, matplotlib 3.10.8.  Results may differ in the last digit
under other versions of scikit-learn (solver tolerances), never in substance.
"""
import sys, time, warnings, csv
import numpy as np
from scipy.special import logit, expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
R_A, R_A2, R_B, R_C = (10, 10, 10, 10) if QUICK else (200, 100, 100, 200)   # replicates per experiment
SEED_A, SEED_B, SEED_C, SEED_A2 = 2026, 2027, 2028, 2029     # one fixed seed per experiment

# ----------------------------------------------------------------------------
# Data-generating process, equation (simdesign) of the manuscript
# ----------------------------------------------------------------------------
P_I, P_E, PREV = 40, 8, 0.30
U_I = np.zeros(P_I); U_I[:10] = 1 / np.sqrt(10)          # shared-nuisance direction (image block)
U_E = np.zeros(P_E); U_E[:4] = 1 / np.sqrt(4)            # shared-nuisance direction (EHR block)
DELTA_I = np.zeros(P_I); DELTA_I[:10] = 0.35; DELTA_I[10:20] = 0.25   # image signal, partly aligned with U_I
DELTA_E = np.zeros(P_E); DELTA_E[4:8] = 0.25                          # EHR signal, orthogonal to U_E


def generate(rng, n, rho, se=1.0, sp=1.0):
    """Return X_I, X_E, true Y, misclassified Y~ (Se, Sp), and the nuisance z."""
    y = (rng.random(n) < PREV).astype(int)
    z = rng.standard_normal(n)
    g = np.sqrt(3 * rho)
    x_i = np.outer(y, DELTA_I) + g * np.outer(z, U_I) + rng.standard_normal((n, P_I))
    x_e = np.outer(y, DELTA_E) + g * np.outer(z, U_E) + rng.standard_normal((n, P_E))
    y_tilde = np.where(y == 1, rng.random(n) < se, rng.random(n) > sp).astype(int)
    return x_i, x_e, y, y_tilde, z


def r_of_rho(rho):
    """Within-class correlation between the shared components: r = 3rho/(1+3rho)."""
    return 3 * rho / (1 + 3 * rho)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def calibration(p, y):
    """Calibration slope and intercept: logistic regression of y on logit(p) (Cox 1958)."""
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lp, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000)


# ----------------------------------------------------------------------------
# Experiment A: cross-modal covariance (five estimators)
# ----------------------------------------------------------------------------
METHODS_A = ["Image only", "Image only (shrinkage LDA)", "Late (Prop. 1)", "Late (logit sum, uncorrected)",
             "Late (naive averaging)", "Early (ridge logistic)", "Joint (shrinkage LDA)"]
METHODS_A2 = ["Image only (tuned)", "Late (Prop. 1, tuned)", "Early (tuned)"]
POP = ["Population: image-only Bayes rule", "Population: late-fusion rule", "Population: Bayes rule"]
RHO_GRID = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
RHO_TAB = [0.0, 0.30, 0.60, 0.90]            # values tabulated in Table 3 and used in Experiment A'


def shrink_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")        # Ledoit-Wolf shrinkage


def fit_all(x_i, x_e, y, x_it, x_et):
    prior = y.mean()
    p_i = lr().fit(x_i, y).predict_proba(x_it)[:, 1]
    p_e = lr().fit(x_e, y).predict_proba(x_et)[:, 1]
    xc, xct = np.hstack([x_i, x_e]), np.hstack([x_it, x_et])
    p_i_lda = shrink_lda().fit(x_i, y).predict_proba(x_it)[:, 1]       # matched unimodal baseline for joint fusion
    late = expit(logit(p_i) + logit(p_e) - logit(prior))               # equation (additivity)
    logit_sum = expit(logit(p_i) + logit(p_e))                         # product of experts: prior counted twice
    naive = 0.5 * (p_i + p_e)                                          # linear opinion pool
    early = lr().fit(xc, y).predict_proba(xct)[:, 1]
    joint = shrink_lda().fit(xc, y).predict_proba(xct)[:, 1]
    return dict(zip(METHODS_A, [p_i, p_i_lda, late, logit_sum, naive, early, joint]))


def population_aucs(rho):
    """Closed-form population AUCs under the Gaussian design (Proposition 2, equation aucloss):
    AUC(b) = Phi{ b'delta / sqrt(2 b'Sigma b) },  Sigma = I + 3 rho w w',  w = (u_I', u_E')'."""
    w, delta = np.concatenate([U_I, U_E]), np.concatenate([DELTA_I, DELTA_E])
    sigma = np.eye(P_I + P_E) + 3 * rho * np.outer(w, w)
    b_i = np.linalg.solve(sigma[:P_I, :P_I], DELTA_I)
    b_e = np.linalg.solve(sigma[P_I:, P_I:], DELTA_E)
    auc = lambda b: float(norm.cdf(b @ delta / np.sqrt(2 * b @ sigma @ b)))
    return dict(zip(POP, [auc(np.concatenate([b_i, np.zeros(P_E)])), auc(np.concatenate([b_i, b_e])),
                          auc(np.linalg.solve(sigma, delta))]))


def experiment_a(rows):
    rng = np.random.default_rng(SEED_A)
    n_train, n_test = 600, 4000
    for rho in RHO_GRID:
        for rep in range(R_A):
            x_i, x_e, y, _, _ = generate(rng, n_train, rho)
            x_it, x_et, yt, _, _ = generate(rng, n_test, rho)
            preds = fit_all(x_i, x_e, y, x_it, x_et)
            for m, p in preds.items():
                slope, intercept = calibration(p, yt)
                rows.append(dict(experiment="A", setting=rho, method=m, rep=rep,
                                 auc=roc_auc_score(yt, p), slope=slope, intercept=intercept,
                                 brier=brier_score_loss(yt, p)))


def experiment_a_tuned(rows):
    """Sensitivity analysis A': every logistic penalty tuned by three-fold CV of the log-likelihood."""
    rng = np.random.default_rng(SEED_A2)
    cv = lambda: LogisticRegressionCV(Cs=[0.03, 0.1, 0.3, 1, 3], cv=3, scoring="neg_log_loss", max_iter=5000)
    for rho in RHO_TAB:
        for rep in range(R_A2):
            x_i, x_e, y, _, _ = generate(rng, 600, rho)
            x_it, x_et, yt, _, _ = generate(rng, 4000, rho)
            p_i = cv().fit(x_i, y).predict_proba(x_it)[:, 1]
            p_e = cv().fit(x_e, y).predict_proba(x_et)[:, 1]
            late = expit(logit(p_i) + logit(p_e) - logit(y.mean()))
            early = cv().fit(np.hstack([x_i, x_e]), y).predict_proba(np.hstack([x_it, x_et]))[:, 1]
            for m, p in zip(METHODS_A2, [p_i, late, early]):
                slope, intercept = calibration(p, yt)
                rows.append(dict(experiment="A2", setting=rho, method=m, rep=rep, auc=roc_auc_score(yt, p),
                                 slope=slope, intercept=intercept, brier=brier_score_loss(yt, p)))


# ----------------------------------------------------------------------------
# Experiment B: outcome misclassification (early fusion, CV-tuned ridge)
# ----------------------------------------------------------------------------
SESP_GRID = [(1.0, 1.0), (0.9, 0.95), (0.8, 0.9), (0.7, 0.85)]


def experiment_b(rows):
    rng = np.random.default_rng(SEED_B)
    rho, n_train, n_test = 0.30, 3000, 4000
    for se, sp in SESP_GRID:
        for rep in range(R_B):
            x_i, x_e, y, y_tilde, _ = generate(rng, n_train, rho, se, sp)
            x_it, x_et, yt, _, _ = generate(rng, n_test, rho)
            model = LogisticRegressionCV(Cs=[0.1, 0.3, 1, 3, 10], cv=3, scoring="neg_log_loss", max_iter=5000)
            p = model.fit(np.hstack([x_i, x_e]), y_tilde).predict_proba(np.hstack([x_it, x_et]))[:, 1]
            # plug-in back-transformation through the inverse of the contraction P(Y~=1|x) = (1-Sp) + (Se+Sp-1) eta(x)
            p_corr = np.clip((p - (1 - sp)) / (se + sp - 1), 1e-4, 1 - 1e-4)
            s, i = calibration(p, yt)
            sc, ic = calibration(p_corr, yt)
            rows.append(dict(experiment="B", setting=f"{se},{sp}", method="Early (CV ridge)", rep=rep,
                             auc=roc_auc_score(yt, p), slope=s, intercept=i, slope_corr=sc, intercept_corr=ic))


# ----------------------------------------------------------------------------
# Experiment C: selection into imaging (early fusion, fixed ridge)
# ----------------------------------------------------------------------------
GAMMA_GRID = [0, 1, 2, 3]


def experiment_c(rows):
    rng = np.random.default_rng(SEED_C)
    rho, n_pool, n_sel, n_test = 0.30, 8000, 1000, 4000
    for gamma in GAMMA_GRID:
        for rep in range(R_C):
            x_i, x_e, y, _, z = generate(rng, n_pool, rho)
            selected = rng.random(n_pool) < expit(-1 + gamma * y + z)   # P(S=1 | y, z)
            idx = np.where(selected)[0][:n_sel]
            x_it, x_et, yt, _, zt = generate(rng, n_test, rho)
            st = rng.random(n_test) < expit(-1 + gamma * yt + zt)        # imaged subset of the target population
            model = lr(C=0.3).fit(np.hstack([x_i[idx], x_e[idx]]), y[idx])
            p = model.predict_proba(np.hstack([x_it, x_et]))[:, 1]
            prev_sel, prev_target = y[idx].mean(), yt.mean()
            # prior-odds correction (equation priorshift) using the known target prevalence
            p_corr = expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) + np.log(prev_target * (1 - prev_sel) / (prev_sel * (1 - prev_target))))
            slope, intercept = calibration(p, yt)
            _, intercept_corr = calibration(p_corr, yt)
            rows.append(dict(experiment="C", setting=gamma, method="Early (ridge logistic)", rep=rep,
                             auc_selected=roc_auc_score(yt[st], p[st]), auc=roc_auc_score(yt, p),
                             prevalence_selected=prev_sel, slope=slope, intercept=intercept, intercept_corr=intercept_corr))


# ----------------------------------------------------------------------------
# Summaries, table and figure
# ----------------------------------------------------------------------------
def agg(vals, stat):
    v = np.asarray(vals, float)
    return (np.mean(v) if stat == "mean" else np.median(v)), np.std(v, ddof=1) / np.sqrt(len(v))


def summarise(rows):
    order = METHODS_A + METHODS_A2
    out = []
    for exp, stat in [("A", "mean"), ("A2", "mean"), ("B", "mean"), ("C", "mean")]:
        sub = [r for r in rows if r["experiment"] == exp]
        keys = sorted({(r["setting"], r["method"]) for r in sub}, key=lambda k: (str(k[0]), order.index(k[1]) if k[1] in order else 0))
        for setting, method in keys:
            grp = [r for r in sub if r["setting"] == setting and r["method"] == method]
            rec = dict(experiment=exp, setting=setting, method=method, n_rep=len(grp), statistic=stat)
            for field in ["auc", "auc_selected", "slope", "intercept", "brier", "slope_corr", "intercept_corr", "prevalence_selected"]:
                if field in grp[0]:
                    m, se = agg([r[field] for r in grp], stat)
                    rec[field], rec[field + "_mcse"] = m, se
            out.append(rec)
    for rho in RHO_GRID:                                                 # closed-form population benchmarks
        for m, v in population_aucs(rho).items():
            out.append(dict(experiment="Apop", setting=rho, method=m, n_rep=0, statistic="closed form", auc=v, auc_mcse=0.0))
    return out


def fmt(x, d=3, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("+", "$+$").replace("-", "$-$")


def _get(summary):
    return lambda exp, setting, method: next(r for r in summary if r["experiment"] == exp and r["setting"] == setting and r["method"] == method)


def write_table(summary, path):
    get = _get(summary)
    span = lambda x: f"\\multicolumn{{2}}{{c}}{{{x}}}"
    pair = lambda r: f"{r['auc']:.3f} & {r['slope']:.3f}"
    head_r = "$r$ & " + " & ".join(span(f"${r_of_rho(c):.3f}$").replace("$0.000$", "$0$") for c in RHO_TAB) + "\\\\"
    rule = "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}"
    names = {"Image only": "Image only, ridge logistic", "Image only (shrinkage LDA)": "Image only, shrinkage LDA",
             "Late (Prop. 1)": "Late, Proposition~\\ref{prop:one}", "Late (logit sum, uncorrected)": "Late, uncorrected logit sum",
             "Late (naive averaging)": "Late, naive averaging", "Early (ridge logistic)": "Early, ridge logistic",
             "Joint (shrinkage LDA)": "Joint, shrinkage LDA", POP[0]: "Image-only Bayes rule",
             POP[1]: "Late-fusion rule, \\eqref{eq:aucloss}", POP[2]: "Bayes rule, $\\AUC^\\ast$",
             "Image only (tuned)": "Image only, ridge logistic", "Late (Prop. 1, tuned)": "Late, Proposition~\\ref{prop:one}",
             "Early (tuned)": "Early, ridge logistic"}
    L = ["\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
         "\\multicolumn{9}{@{}l}{\\textbf{A. Cross-modal covariance} ($n=600$; mean AUC / calibration slope, 200 replicates)}\\\\", head_r, rule]
    L += [names[m] + " & " + " & ".join(pair(get("A", c, m)) for c in RHO_TAB) + "\\\\" for m in METHODS_A
          if m != "Late (logit sum, uncorrected)"]   # same AUC and slope as Prop. 1 by construction; intercepts in CSV
    L.append("\\multicolumn{9}{@{}l}{\\emph{Population values from \\eqref{eq:aucloss} (closed form; AUC only)}}\\\\")
    L += [names[m] + " & " + " & ".join(span(f"{get('Apop', c, m)['auc']:.3f}") for c in RHO_TAB) + "\\\\" for m in POP]
    L += ["\\midrule", "\\multicolumn{9}{@{}l}{\\textbf{A$'$. Penalties tuned by cross-validation} ($n=600$; 100 replicates)}\\\\", head_r, rule]
    L += [names[m] + " & " + " & ".join(pair(get("A2", c, m)) for c in RHO_TAB) + "\\\\" for m in METHODS_A2]
    L += ["\\midrule", "\\multicolumn{9}{@{}l}{\\textbf{B. Label misclassification} ($n=3000$, $r=0.474$; means, 100 replicates)}\\\\"]
    b = [get("B", f"{se},{sp}", "Early (CV ridge)") for se, sp in SESP_GRID]
    L.append("$(\\Se,\\Sp)$ & " + " & ".join(span(f"$({se:g},{sp:g})$") for se, sp in SESP_GRID) + "\\\\")
    L.append("AUC against true $Y$ & " + " & ".join(span(f"{r['auc']:.3f}") for r in b) + "\\\\")
    L.append("Calibration slope / intercept & " + " & ".join(span(f"{r['slope']:.3f} / {fmt(r['intercept'], 3, True)}") for r in b) + "\\\\")
    L.append("Corrected slope / intercept & " + " & ".join(span(f"{r['slope_corr']:.3f} / {fmt(r['intercept_corr'], 3, True)}") for r in b) + "\\\\")
    L += ["\\midrule", "\\multicolumn{9}{@{}l}{\\textbf{C. Selection into imaging} ($n=1000$, $r=0.474$; means, 200 replicates)}\\\\"]
    c = [get("C", g, "Early (ridge logistic)") for g in GAMMA_GRID]
    L.append("$\\gamma$ & " + " & ".join(span(f"${g}$") for g in GAMMA_GRID) + "\\\\")
    L.append("Prevalence in imaged cohort & " + " & ".join(span(f"{r['prevalence_selected']:.3f}") for r in c) + "\\\\")
    L.append("AUC, imaged / target & " + " & ".join(span(f"{r['auc_selected']:.3f} / {r['auc']:.3f}") for r in c) + "\\\\")
    L.append("Calibration slope in target & " + " & ".join(span(f"{r['slope']:.3f}") for r in c) + "\\\\")
    L.append("Calibration intercept in target & " + " & ".join(span(fmt(r['intercept'], 3, True)) for r in c) + "\\\\")
    L.append("Corrected intercept & " + " & ".join(span(fmt(r['intercept_corr'], 3, True)) for r in c) + "\\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(L) + "\n")


def figure_data(summary):
    get = _get(summary)
    rs = [r_of_rho(rho) for rho in RHO_GRID]
    base = {"Late (Prop. 1)": "Image only", "Early (ridge logistic)": "Image only",
            "Joint (shrinkage LDA)": "Image only (shrinkage LDA)"}          # matched unimodal comparators
    gain = {m: [get("A", rho, m)["auc"] - get("A", rho, b)["auc"] for rho in RHO_GRID] for m, b in base.items()}
    popgain = {k: [get("Apop", rho, v)["auc"] - get("Apop", rho, POP[0])["auc"] for rho in RHO_GRID]
               for k, v in {"Population: Bayes rule": POP[2], "Population: late-fusion rule": POP[1]}.items()}
    slope = {m: [get("A", rho, m)["slope"] for rho in RHO_GRID]
             for m in ["Late (naive averaging)", "Late (Prop. 1)", "Joint (shrinkage LDA)"]}
    return rs, gain, popgain, slope


def write_pgfplots(rs, gain, popgain, slope, path):
    """Write the complete tikzpicture of Figure 4 (identical in the manuscript and in figure4_simulation.tex)."""
    co = lambda ys: "".join(f"({r:.3f},{y:.3f})" for r, y in zip(rs, ys))
    ymax = np.ceil(100 * 1.55 * max(max(v) for v in list(gain.values()) + list(popgain.values()))) / 100
    A = [("mark=o,thick", gain["Late (Prop. 1)"], "late fusion (Prop.~1)"),
         ("mark=square,thick,dashed", gain["Early (ridge logistic)"], "early fusion (ridge logistic)"),
         ("mark=triangle,thick,densely dotted", gain["Joint (shrinkage LDA)"], "joint fusion (shrinkage LDA)"),
         ("gray,thick", popgain["Population: Bayes rule"], "population: Bayes rule"),
         ("gray,thick,dashed", popgain["Population: late-fusion rule"], "population: late-fusion rule")]
    B = [("mark=diamond,thick", slope["Late (naive averaging)"], "naive probability averaging"),
         ("mark=o,thick", slope["Late (Prop. 1)"], "late fusion (Prop.~1)"),
         ("mark=triangle,thick,densely dotted", slope["Joint (shrinkage LDA)"], "joint fusion (shrinkage LDA)")]
    L = ["\\begin{tikzpicture}",
         "\\begin{axis}[name=A,width=7.6cm,height=5.8cm,xlabel={within-class cross-modal correlation $r$},ylabel={$\\Delta$AUC vs.\\ matched image-only},",
         " scaled y ticks=false,yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},",
         f" legend style={{font=\\scriptsize,at={{(0.03,0.97)}},anchor=north west,draw=none}},grid=major,ymin=0,ymax={ymax:.2f},xmin=0,xmax=0.75,title={{\\small (a) Discrimination gain from the clinical block}}]"]
    for sty, ys, lab in A:
        L += [f"\\addplot[{sty}] coordinates {{{co(ys)}}};", f"\\addlegendentry{{{lab}}}"]
    L += ["\\end{axis}",
          "\\begin{axis}[at={(A.east)},anchor=west,xshift=1.6cm,width=7.6cm,height=5.8cm,xlabel={within-class cross-modal correlation $r$},ylabel={calibration slope},",
          " legend style={font=\\scriptsize,at={(0.97,0.97)},anchor=north east,draw=none},grid=major,ymin=0.5,ymax=2.4,xmin=0,xmax=0.75,title={\\small (b) Calibration of the fused probabilities}]"]
    for sty, ys, lab in B:
        L += [f"\\addplot[{sty}] coordinates {{{co(ys)}}};", f"\\addlegendentry{{{lab}}}"]
    L += ["\\addplot[gray,dashed,domain=0:0.75] {1};", "\\end{axis}", "\\end{tikzpicture}"]
    open(path, "w").write("% Figure 4 tikzpicture, generated by simulation_section4_7.py\n" + "\n".join(L) + "\n")


def write_figure(rs, gain, popgain, slope, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = {"Late (Prop. 1)": ("o", "-", "k", "late fusion (Prop. 1)"), "Early (ridge logistic)": ("s", "--", "k", "early fusion (ridge logistic)"),
           "Joint (shrinkage LDA)": ("^", ":", "k", "joint fusion (shrinkage LDA)"), "Late (naive averaging)": ("D", "-", "k", "naive probability averaging"),
           "Population: Bayes rule": (None, "-", "0.6", "population: Bayes rule"), "Population: late-fusion rule": (None, "--", "0.6", "population: late-fusion rule")}
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for m, ys in list(gain.items()) + list(popgain.items()):
        mk, ls, col, lab = sty[m]; ax[0].plot(rs, ys, marker=mk, linestyle=ls, color=col, label=lab)
    ax[0].set(xlabel="within-class cross-modal correlation $r$", ylabel="$\\Delta$AUC relative to matched image-only model",
              title="(a) Discrimination gain from the clinical block", xlim=(0, 0.75), ylim=(0, None))
    for m, ys in slope.items():
        mk, ls, col, lab = sty[m]; ax[1].plot(rs, ys, marker=mk, linestyle=ls, color=col, label=lab)
    ax[1].axhline(1, color="gray", linestyle="--", lw=1)
    ax[1].set(xlabel="within-class cross-modal correlation $r$", ylabel="calibration slope",
              title="(b) Calibration of the fused probabilities", xlim=(0, 0.75), ylim=(0.5, 2.4))
    for a in ax:
        a.legend(frameon=False, fontsize=7); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path_stem + ".pdf"); fig.savefig(path_stem + ".png", dpi=200)


def write_csv(rows, path):
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("experiment", "setting", "method", "rep", "n_rep", "statistic"), k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    t0 = time.time(); rows = []
    for name, fn in [("A", experiment_a), ("A'", experiment_a_tuned), ("B", experiment_b), ("C", experiment_c)]:
        fn(rows); print(f"experiment {name} done  ({time.time() - t0:.0f} s)", flush=True)
    summary = summarise(rows)
    write_csv(rows, "simulation_results_long.csv")
    write_csv(summary, "simulation_results_summary.csv")
    write_table(summary, "table_simulation.tex")
    rs, gain, popgain, slope = figure_data(summary)
    write_pgfplots(rs, gain, popgain, slope, "figure_simulation_pgfplots.tex")
    write_figure(rs, gain, popgain, slope, "figure_simulation")
    for r in summary:
        keys = [k for k in ("auc", "auc_selected", "slope", "intercept", "slope_corr", "intercept_corr", "prevalence_selected", "brier") if k in r]
        print(f"{r['experiment']} {str(r['setting']):8s} {r['method']:24s} " + "  ".join(f"{k}={r[k]:.3f}" for k in keys))
    print(f"total time {time.time() - t0:.0f} s")
