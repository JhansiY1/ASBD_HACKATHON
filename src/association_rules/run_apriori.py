"""
PHASE 7 - Association Rule Mining with the from-scratch Apriori.

1. load film transactions (data/transactions/film_transactions.json)
2. sensitivity grid over min_support x min_confidence
3. final run with the selected thresholds
4. validation against mlxtend's independent Apriori implementation
5. rule filtering / ranking, interpretation table and plots

Run:  python -m src.association_rules.run_apriori
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.association_rules.apriori import apriori, generate_rules
from src.config import (APRIORI_DIR, APRIORI_MIN_CONFIDENCE, APRIORI_MIN_SUPPORT, PLOTS_DIR, TRANSACTIONS_DIR,
                        get_logger)
from src.visualization.style import CAT, GREY, INK2, SEQ, apply_style, save

log = get_logger("apriori")
SUPPORTS = [0.01, 0.02, 0.03, 0.05]
CONFIDENCES = [0.50, 0.60, 0.70, 0.80]


def load_transactions() -> list[list[str]]:
    with open(TRANSACTIONS_DIR / "film_transactions.json", encoding="utf-8") as fh:
        return json.load(fh)


def sensitivity(tx) -> pd.DataFrame:
    rows = []
    for s in SUPPORTS:
        res = apriori(tx, s)
        for c in CONFIDENCES:
            rules = generate_rules(res, c)
            rows.append(dict(min_support=s, min_confidence=c, frequent_itemsets=len(res.itemsets),
                             max_itemset_len=max(len(i) for i in res.itemsets), rules=len(rules),
                             rules_lift_gt_1_2=int((rules["lift"] > 1.2).sum()),
                             median_lift=float(rules["lift"].median()) if len(rules) else np.nan,
                             runtime_s=round(res.runtime_s, 4)))
    return pd.DataFrame(rows)


def validate_with_mlxtend(tx, min_support) -> dict:
    """Independent check: mlxtend's Apriori must give identical itemsets & supports."""
    from mlxtend.frequent_patterns import apriori as mx_apriori
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    X = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    t0 = time.perf_counter()
    mx = mx_apriori(X, min_support=min_support, use_colnames=True)
    t_mx = time.perf_counter() - t0
    ours = apriori(tx, min_support)
    mx_map = {frozenset(s): round(v * len(tx)) for s, v in zip(mx["itemsets"], mx["support"])}
    same_sets = set(mx_map) == set(ours.itemsets)
    same_counts = same_sets and all(mx_map[k] == v for k, v in ours.itemsets.items())
    return dict(mlxtend_itemsets=len(mx_map), our_itemsets=len(ours.itemsets), identical_itemsets=same_sets,
                identical_support_counts=same_counts, mlxtend_runtime_s=round(t_mx, 4),
                our_runtime_s=round(ours.runtime_s, 4))


def closure(items: frozenset, tidsets: dict[str, set], all_items: list[str]) -> frozenset:
    """Closure operator f(X) = i(t(X)) of Lecture 11 (A-Close): the set of
    items common to every transaction that contains X."""
    tids = set.intersection(*(tidsets[i] for i in items))
    return frozenset(i for i in all_items if tids <= tidsets[i])


def select_meaningful(rules: pd.DataFrame, tx: list[list[str]], top_n: int = 20) -> pd.DataFrame:
    """Rank rules for interpretation without combinatorial repetition.

    * single-item consequents (easiest to read and to act on),
    * lift > 1.1 (positive dependence; lift ~1 rules only restate base rates),
    * redundancy removal with the closure operator: all rules whose itemset
      X u Y has the same closure i(t(X u Y)) describe exactly the same set of
      films; only the best rule (highest lift, then shortest antecedent) of
      each closure group is kept,
    * at most two rules per consequent, ordered by lift then confidence.
    """
    tidsets: dict[str, set] = {}
    for tid, t in enumerate(tx):
        for i in t:
            tidsets.setdefault(i, set()).add(tid)
    all_items = sorted(tidsets)
    r = rules[(rules["consequent_len"] == 1) & (rules["lift"] > 1.1)].copy()
    r["itemset"] = [frozenset(a.split(", ")) | frozenset([c]) for a, c in zip(r["antecedent"], r["consequent"])]
    r["closure"] = [closure(s, tidsets, all_items) for s in r["itemset"]]
    r = r.sort_values(["lift", "confidence", "antecedent_len"], ascending=[False, False, True])
    r = r.drop_duplicates(subset="closure", keep="first")
    r = r.groupby("consequent", sort=False).head(3)
    return r.drop(columns=["itemset", "closure"]).head(top_n).reset_index(drop=True)


def rare_quality_rules(tx, min_support=0.01, min_conf=0.15) -> pd.DataFrame:
    """Supplementary run: critic-rating / Oscar items are rare (75 % of films
    are not in Halliwell's guide, each star level covers 1-9 %), so rules
    towards them only surface below the main 2 % support threshold and with
    low confidence.  Rules with a rating/Oscar consequent and lift > 2 are
    reported separately as *high-lift, low-confidence* descriptive patterns."""
    res = apriori(tx, min_support)
    rules = generate_rules(res, min_conf)
    r = rules[(rules["consequent_len"] == 1) & rules["consequent"].str.contains("halliwell=|oscar=")
              & (rules["lift"] > 2.0)]
    r = select_meaningful(r, tx, 15)
    return r


def interpret(row) -> str:
    return (f"Films with {{{row.antecedent}}} have {{{row.consequent}}} in {100 * row.confidence:.0f}% of cases, "
            f"{row.lift:.2f}x the base rate of {100 * row.confidence / row.lift:.1f}% "
            f"(support {100 * row.support:.1f}%, {int(row.support_count)} films).")


def plots(rules, sens, levels):
    apply_style()
    # (a) support-confidence scatter coloured by lift
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap=SEQ, s=26 + 6 * rules["antecedent_len"],
                    edgecolor="white", linewidth=0.5)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Lift")
    ax.set_title(f"Association rules: support vs confidence (colour = lift), n = {len(rules)}")
    ax.set_xlabel("Support"); ax.set_ylabel("Confidence")
    ax.set_xscale("log")
    p1 = save(fig, PLOTS_DIR / "17_apriori_support_confidence_lift.png")
    # (b) sensitivity: frequent itemsets & rules vs thresholds
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fi = sens.groupby("min_support")["frequent_itemsets"].first()
    axes[0].plot(fi.index, fi.values, marker="o", color=CAT[0])
    for x, y in fi.items():
        axes[0].annotate(f"{y}", (x, y), xytext=(4, 6), textcoords="offset points", fontsize=9, color=INK2)
    axes[0].set_title("Frequent itemsets vs minimum support")
    axes[0].set_xlabel("Minimum support"); axes[0].set_ylabel("# frequent itemsets")
    for i, c in enumerate(CONFIDENCES):
        d = sens[sens["min_confidence"] == c]
        axes[1].plot(d["min_support"], d["rules"], marker="o", color=CAT[i], label=f"min conf {c:.2f}")
    axes[1].set_title("Number of rules vs thresholds")
    axes[1].set_xlabel("Minimum support"); axes[1].set_ylabel("# rules"); axes[1].legend()
    p2 = save(fig, PLOTS_DIR / "18_apriori_threshold_sensitivity.png")
    # (c) level-wise candidate generation and pruning
    lv = pd.DataFrame([l.__dict__ for l in levels])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.27
    x = np.arange(len(lv))
    ax.bar(x - w, lv["candidates_joined"], w, color=GREY, label="candidates after join")
    ax.bar(x, lv["candidates_counted"], w, color=CAT[0], label="candidates after prune (counted)")
    ax.bar(x + w, lv["frequent"], w, color=CAT[1], label="frequent itemsets L_k")
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in lv["k"]])
    ax.set_yscale("log")
    ax.set_title("Apriori level-wise search: join, prune and frequent itemsets")
    ax.set_xlabel("Itemset size k"); ax.set_ylabel("Count (log scale)"); ax.legend()
    p3 = save(fig, PLOTS_DIR / "19_apriori_levelwise_candidates.png")
    return [p1.name, p2.name, p3.name]


def top_rules_plot(top):
    apply_style()
    t = top.head(15).iloc[::-1]
    labels = [f"{a}  ->  {c}" for a, c in zip(t["antecedent"], t["consequent"])]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(range(len(t)), t["lift"], color=CAT[0], height=0.65)
    for i, (l, cf) in enumerate(zip(t["lift"], t["confidence"])):
        ax.text(l + 0.05, i, f"lift {l:.2f} | conf {cf:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xlim(0, t["lift"].max() * 1.3)
    ax.set_title("Top association rules ranked by lift")
    ax.set_xlabel("Lift"); ax.set_ylabel("Rule (antecedent -> consequent)")
    return save(fig, PLOTS_DIR / "20_apriori_top_rules_lift.png").name


def run() -> dict:
    log.info("PHASE 7 - Apriori started")
    tx = load_transactions()
    sens = sensitivity(tx)
    sens.to_csv(APRIORI_DIR / "support_analysis.csv", index=False)
    res = apriori(tx, APRIORI_MIN_SUPPORT)
    fi = res.to_frame()
    fi.to_csv(APRIORI_DIR / "frequent_itemsets.csv", index=False)
    pd.DataFrame([l.__dict__ for l in res.levels]).to_csv(APRIORI_DIR / "levelwise_trace.csv", index=False)
    rules = generate_rules(res, APRIORI_MIN_CONFIDENCE)
    rules.to_csv(APRIORI_DIR / "association_rules.csv", index=False)
    top = select_meaningful(rules, tx, 20)
    top["interpretation"] = [interpret(r) for r in top.itertuples()]
    top.to_csv(APRIORI_DIR / "top_rules.csv", index=False)
    rare = rare_quality_rules(tx)
    rare["interpretation"] = [interpret(r) for r in rare.itertuples()]
    rare.to_csv(APRIORI_DIR / "supplementary_rating_oscar_rules.csv", index=False)
    val = validate_with_mlxtend(tx, APRIORI_MIN_SUPPORT)
    # item supports (1-itemsets) for the report
    one = fi[fi["length"] == 1].sort_values("support", ascending=False)
    one.to_csv(APRIORI_DIR / "item_supports.csv", index=False)
    figs = plots(rules, sens, res.levels) + [top_rules_plot(top)]
    summary = dict(n_transactions=len(tx), n_items=len({i for t in tx for i in t}),
                   avg_transaction_length=float(np.mean([len(t) for t in tx])),
                   min_support=APRIORI_MIN_SUPPORT, min_count=res.min_count, min_confidence=APRIORI_MIN_CONFIDENCE,
                   frequent_itemsets=len(res.itemsets), by_length=fi["length"].value_counts().sort_index().to_dict(),
                   rules=len(rules), rules_lift_gt_1=int((rules["lift"] > 1).sum()), top_rules=len(top),
                   runtime_s=res.runtime_s, validation=val, figures=figs)
    with open(APRIORI_DIR / "apriori_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("PHASE 7 complete: %s", summary)
    return summary


if __name__ == "__main__":
    run()
