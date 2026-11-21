"""
PHASE 9 - Closed frequent itemset mining with COFI + validation.

Validation performed:
  V1 lecture example (Lecture 11, A-Close): closed sets must be exactly
     {A,C}, {B,E}, {B,C,E}, {C} at min support count 3
  V2 brute-force exhaustive enumeration on 30 random small databases:
     COFI frequent itemsets and supports identical, closed sets identical
  V3 full movie data: COFI frequent itemsets identical to the from-scratch
     Apriori and to mlxtend FP-Growth (independent implementations)
  V4 every reported closed itemset: support recounted from the raw
     transactions (tidset intersection) >= min_count and equal to the
     reported value; closure operator f(X) = i(t(X)) = X (no equal-support
     proper superset)
  V5 completeness: every frequent itemset's support equals the support of
     its closure, which is among the reported closed itemsets

Run:  python -m src.cofi.run_cofi
"""
from __future__ import annotations

import json
import random
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.association_rules.apriori import apriori
from src.association_rules.run_apriori import load_transactions
from src.cofi.cofi import (brute_force_frequent, closed_and_maximal, closure_operator, cofi_mine,
                           min_count_for, tidsets_of)
from src.config import COFI_DIR, COFI_MIN_SUPPORT, PLOTS_DIR, RANDOM_SEED, get_logger
from src.visualization.style import CAT, GREY, INK2, apply_style, save

log = get_logger("cofi")

LECTURE11_DB = [["A", "C"], ["B", "C", "E"], ["A", "B", "C", "E"], ["B", "E"], ["A", "B", "C", "E"]]


def v1_lecture_example() -> dict:
    res = cofi_mine(LECTURE11_DB, min_count=3)
    got = {tuple(sorted(s)) for s in res.closed}
    expected = {("A", "C"), ("B", "E"), ("B", "C", "E"), ("C",)}
    return dict(closed_found=sorted(got), expected=sorted(expected), passed=got == expected)


def v2_random_bruteforce(n_db=30) -> dict:
    rng = random.Random(RANDOM_SEED)
    ok_f = ok_c = 0
    for _ in range(n_db):
        items = [f"i{j}" for j in range(rng.randint(5, 9))]
        db = [rng.sample(items, rng.randint(1, len(items))) for _ in range(rng.randint(8, 40))]
        mc = rng.randint(2, 5)
        res = cofi_mine(db, min_count=mc)
        bf = brute_force_frequent(db, mc)
        ok_f += res.frequent == bf
        bf_closed, _ = closed_and_maximal(bf, sorted({i for t in db for i in t}))
        ts = tidsets_of(db)
        # closed by the closure-operator definition, computed independently
        bf_closed2 = {x: s for x, s in bf.items() if closure_operator(x, ts) == x}
        ok_c += (res.closed == bf_closed == bf_closed2)
    return dict(databases=n_db, frequent_identical=ok_f, closed_identical=ok_c, passed=ok_f == ok_c == n_db)


def v3_movie_cross_check(tx, res) -> dict:
    ap = apriori(tx, COFI_MIN_SUPPORT)
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    X = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    fg = fpgrowth(X, min_support=COFI_MIN_SUPPORT, use_colnames=True)
    fg_map = {frozenset(s): int(round(v * len(tx))) for s, v in zip(fg["itemsets"], fg["support"])}
    return dict(cofi_frequent=len(res.frequent), apriori_frequent=len(ap.itemsets), mlxtend_fpgrowth_frequent=len(fg_map),
                identical_to_apriori=res.frequent == ap.itemsets, identical_to_mlxtend_fpgrowth=res.frequent == fg_map)


def v4_v5_closed_checks(tx, res) -> dict:
    ts = tidsets_of(tx)
    bad_support = bad_freq = bad_closure = 0
    for x, s in res.closed.items():
        true = len(set.intersection(*(ts[i] for i in x)))
        bad_support += true != s
        bad_freq += true < res.min_count
        bad_closure += closure_operator(x, ts) != x
    missing = 0
    for x, s in res.frequent.items():
        c = closure_operator(x, ts)
        if res.closed.get(c) != s:
            missing += 1
    return dict(closed_checked=len(res.closed), wrong_support=bad_support, infrequent=bad_freq,
                not_closed_by_closure_operator=bad_closure, frequent_without_closed_cover=missing,
                passed=bad_support == bad_freq == bad_closure == missing == 0)


def run() -> dict:
    log.info("PHASE 9 - COFI started")
    tx = load_transactions()
    tracemalloc.start()
    t0 = time.perf_counter()
    res = cofi_mine(tx, COFI_MIN_SUPPORT)
    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    v = dict(V1_lecture11_example=v1_lecture_example(), V2_bruteforce_random=v2_random_bruteforce(),
             V3_movie_cross_check=v3_movie_cross_check(tx, res), V4_V5_closed_checks=v4_v5_closed_checks(tx, res))
    n = len(tx)
    rows = [dict(itemset=", ".join(sorted(x)), support_count=s, support_pct=round(100 * s / n, 3), n_items=len(x),
                 closed=True, maximal=x in res.maximal) for x, s in res.closed.items()]
    cfi = pd.DataFrame(rows).sort_values(["n_items", "support_count"], ascending=[False, False])
    cfi.to_csv(COFI_DIR / "closed_frequent_itemsets.csv", index=False)
    pd.DataFrame([s.__dict__ for s in res.tree_stats]).to_csv(COFI_DIR / "cofi_tree_statistics.csv", index=False)

    # support comparison over thresholds: frequent vs closed vs maximal
    comp = []
    for ms in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
        t1 = time.perf_counter()
        r = cofi_mine(tx, ms)
        dt = time.perf_counter() - t1
        t2 = time.perf_counter()
        ap = apriori(tx, ms)
        dta = time.perf_counter() - t2
        comp.append(dict(min_support=ms, min_count=r.min_count, frequent_itemsets=len(r.frequent),
                         closed_itemsets=len(r.closed), maximal_itemsets=len(r.maximal),
                         compression_ratio=round(len(r.frequent) / max(1, len(r.closed)), 3),
                         cofi_time_s=round(dt, 4), apriori_time_s=round(dta, 4),
                         fp_tree_nodes=r.fp_tree_nodes, largest_cofi_tree_nodes=r.max_cofi_nodes,
                         identical_to_apriori=r.frequent == ap.itemsets))
    comp = pd.DataFrame(comp)
    comp.to_csv(COFI_DIR / "support_comparison.csv", index=False)

    # per-itemset comparison at the chosen threshold (frequent vs closed status)
    per = pd.DataFrame([dict(itemset=", ".join(sorted(x)), support_count=s, n_items=len(x),
                             closed=x in res.closed, maximal=x in res.maximal) for x, s in res.frequent.items()])
    per.sort_values(["n_items", "support_count"], ascending=[True, False]).to_csv(
        COFI_DIR / "frequent_itemsets_closed_status.csv", index=False)

    figs = plots(comp, res)
    summary = dict(n_transactions=n, min_support=COFI_MIN_SUPPORT, min_count=res.min_count,
                   frequent_items=len(res.tree_stats), fp_tree_nodes=res.fp_tree_nodes,
                   largest_cofi_tree_nodes=res.max_cofi_nodes, frequent_itemsets=len(res.frequent),
                   closed_itemsets=len(res.closed), maximal_itemsets=len(res.maximal),
                   time_fp_tree_s=res.time_fp_tree_s, time_cofi_mining_s=res.time_cofi_mining_s,
                   time_closure_check_s=res.time_closure_s, total_time_s=total, peak_python_memory_mb=peak / 1e6,
                   validation=v, figures=figs)
    write_summary(summary, cfi, comp)
    with open(COFI_DIR / "cofi_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("PHASE 9 complete: frequent=%d closed=%d maximal=%d validation=%s", len(res.frequent),
             len(res.closed), len(res.maximal), {k: v_.get("passed", v_) for k, v_ in v.items()})
    return summary


def plots(comp, res):
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for col, c, lab in [("frequent_itemsets", CAT[0], "all frequent itemsets"),
                        ("closed_itemsets", CAT[1], "closed frequent itemsets (COFI)"),
                        ("maximal_itemsets", CAT[2], "maximal frequent itemsets")]:
        ax.plot(comp["min_support"], comp[col], marker="o", color=c, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Frequent vs closed vs maximal itemsets across support thresholds")
    ax.set_xlabel("Minimum support (log scale)"); ax.set_ylabel("Number of itemsets (log scale)")
    ax.legend()
    p1 = save(fig, PLOTS_DIR / "21_cofi_frequent_closed_maximal.png")
    st = pd.DataFrame([s.__dict__ for s in res.tree_stats])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(st["support"], st["nodes"], s=20 + st["frequent_patterns"] * 2, color=CAT[0], alpha=0.75,
               edgecolor="white")
    for _, r in st.nlargest(5, "nodes").iterrows():
        ax.annotate(r["item"], (r["support"], r["nodes"]), fontsize=8, color=INK2, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_title("Size of each item's COFI-tree vs item support (bubble = patterns mined)")
    ax.set_xlabel("Support count of the COFI-tree root item"); ax.set_ylabel("COFI-tree nodes")
    p2 = save(fig, PLOTS_DIR / "22_cofi_tree_sizes.png")
    return [p1.name, p2.name]


def write_summary(s, cfi, comp):
    v = s["validation"]
    lines = ["COFI CLOSED FREQUENT ITEMSET MINING - EXECUTION SUMMARY", "=" * 64,
             f"Transactions             : {s['n_transactions']}",
             f"Minimum support          : {s['min_support']} (count >= {s['min_count']})",
             f"Frequent items (COFI-trees built): {s['frequent_items']}",
             f"Global FP-tree nodes     : {s['fp_tree_nodes']}",
             f"Largest single COFI-tree : {s['largest_cofi_tree_nodes']} nodes (only one tree in memory at a time)",
             f"Frequent itemsets        : {s['frequent_itemsets']}",
             f"Closed frequent itemsets : {s['closed_itemsets']}",
             f"Maximal frequent itemsets: {s['maximal_itemsets']}",
             f"Compression (frequent/closed): {s['frequent_itemsets'] / s['closed_itemsets']:.2f}x",
             f"Time: FP-tree {s['time_fp_tree_s']:.4f}s, COFI mining {s['time_cofi_mining_s']:.4f}s, "
             f"closedness check {s['time_closure_check_s']:.4f}s, total {s['total_time_s']:.4f}s",
             f"Peak Python heap during mining: {s['peak_python_memory_mb']:.2f} MB", "",
             "VALIDATION", "-" * 64]
    for k, val in v.items():
        lines.append(f"{k}: {json.dumps(val, default=str)}")
    lines += ["", "SUPPORT COMPARISON", "-" * 64, comp.to_string(index=False), "",
              "LARGEST CLOSED ITEMSETS", "-" * 64, cfi.head(15).to_string(index=False)]
    (COFI_DIR / "cofi_execution_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
