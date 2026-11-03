# -*- coding: utf-8 -*-
"""
Statistical comparison of the six training configurations on the VALIDATION subset.

Input : a CSV with one row per (configuration, seed), containing at least the columns
        config_id, seed, mAP50_mask, mAP50_95_mask   (e.g. results/ablation_validation_runs.csv)
Tests : - one-way ANOVA (configuration as factor)
        - randomized-block ANOVA (configuration + seed as block, no interaction), because
          all stage-2 models of a seed share the same stage-1 model
        - Friedman test (non-parametric, seed as block)
        - Welch t-tests between the best configuration and every other one (exploratory)

Usage : python scripts/analysis/ablation_statistics.py --csv results/ablation_validation_runs.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

ORDER = ["single_raw", "ft_raw_raw", "single_proc", "ft_proc_proc", "cross_raw_proc", "cross_proc_raw"]


def block_anova(df, metric):
    """Two-way ANOVA without interaction: y = config + seed + error."""
    y = df[metric].values
    gm = y.mean()
    a, b = df["config_id"].nunique(), df["seed"].nunique()
    ss_tot = ((y - gm) ** 2).sum()
    ss_cfg = b * ((df.groupby("config_id")[metric].mean() - gm) ** 2).sum()
    ss_seed = a * ((df.groupby("seed")[metric].mean() - gm) ** 2).sum()
    ss_err = ss_tot - ss_cfg - ss_seed
    df_cfg, df_seed, df_err = a - 1, b - 1, (a - 1) * (b - 1)
    ms_err = ss_err / df_err
    f_cfg, f_seed = (ss_cfg / df_cfg) / ms_err, (ss_seed / df_seed) / ms_err
    return dict(F_config=f_cfg, df_config=(df_cfg, df_err), p_config=stats.f.sf(f_cfg, df_cfg, df_err),
                F_seed=f_seed, df_seed=(df_seed, df_err), p_seed=stats.f.sf(f_seed, df_seed, df_err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metrics", nargs="+", default=["mAP50_95_mask", "mAP50_mask"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["config_id"].isin(ORDER)].copy()
    n = df.groupby("config_id").size()
    if (n != n.iloc[0]).any():
        raise SystemExit(f"Unbalanced design (runs per configuration): {n.to_dict()}")

    for m in args.metrics:
        print(f"\n================  {m}  ================")
        summ = df.groupby("config_id")[m].agg(["mean", "std", "count"]).reindex(ORDER)
        print(summ.to_string(float_format=lambda v: f"{v:.4f}"))

        groups = [df.loc[df.config_id == c, m].values for c in ORDER]
        f, p = stats.f_oneway(*groups)
        k, N = len(groups), sum(len(g) for g in groups)
        print(f"\nOne-way ANOVA            : F({k-1},{N-k}) = {f:.2f}, p = {p:.3f}")

        b = block_anova(df, m)
        print(f"Randomized-block ANOVA   : configuration F{b['df_config']} = {b['F_config']:.2f}, "
              f"p = {b['p_config']:.3f};  seed F{b['df_seed']} = {b['F_seed']:.2f}, p = {b['p_seed']:.3f}")

        wide = df.pivot(index="seed", columns="config_id", values=m)[ORDER]
        chi2, pf = stats.friedmanchisquare(*[wide[c].values for c in ORDER])
        print(f"Friedman test            : chi2({k-1}) = {chi2:.2f}, p = {pf:.3f}")

        best = summ["mean"].idxmax()
        print(f"\nWelch t-tests vs highest-mean configuration ({best}), exploratory, uncorrected:")
        for c in ORDER:
            if c == best:
                continue
            t, pw = stats.ttest_ind(df.loc[df.config_id == best, m], df.loc[df.config_id == c, m], equal_var=False)
            print(f"  {best} vs {c:<15}: t = {t:6.2f}, p = {pw:.3f}")


if __name__ == "__main__":
    main()
