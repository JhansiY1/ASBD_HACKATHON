"""
Apriori algorithm implemented from scratch (Agrawal & Srikant, 1994;
Lecture 6 "Apriori & FP-Growth").

Input : list of transactions (each a list of item strings), min_support
        (fraction of transactions), min_confidence.
Output: frequent itemsets with support counts; association rules with
        support, confidence, lift, leverage, conviction and coverage.

Pseudo-code followed (Lecture 6, slide 4):
    L1 = {frequent items}
    for (k = 1; Lk != {}; k++):
        C(k+1) = candidates generated from Lk        (self-join + prune)
        for each transaction t: increment count of all candidates in C(k+1) contained in t
        L(k+1) = candidates in C(k+1) with count >= min_support
    return union of Lk

Candidate generation (Lecture 6, slide 5):
    join : two k-itemsets (sorted) that share their first k-1 items are
           merged into one (k+1)-candidate, e.g. abc + abd -> abcd
    prune: a candidate is removed if any of its k-subsets is not in Lk
           (Apriori / downward-closure property: every non-empty subset of
           a frequent itemset is frequent; equivalently, anti-monotonicity -
           a superset of an infrequent itemset is infrequent)

Rule measures, for a rule X -> Y with N transactions:
    support(X -> Y)   = count(X u Y) / N
    confidence        = support(X u Y) / support(X)        = P(Y | X)
    lift              = confidence / support(Y)            (> 1 positive dependence)
    leverage          = support(X u Y) - support(X) * support(Y)
    conviction        = (1 - support(Y)) / (1 - confidence)  (inf if confidence = 1)
    coverage          = support(X)                         (antecedent support)
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd


@dataclass
class LevelStats:
    k: int
    candidates_joined: int     # |C_k| after self-join, before pruning
    pruned: int                # candidates removed by the Apriori property
    candidates_counted: int    # |C_k| after pruning (support counted in DB scan)
    frequent: int              # |L_k|


@dataclass
class AprioriResult:
    n_transactions: int
    min_support: float
    min_count: int
    itemsets: dict[frozenset, int]
    levels: list[LevelStats] = field(default_factory=list)
    runtime_s: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        rows = [dict(itemset=", ".join(sorted(s)), length=len(s), support_count=c,
                     support=c / self.n_transactions) for s, c in self.itemsets.items()]
        return pd.DataFrame(rows).sort_values(["length", "support"], ascending=[True, False]).reset_index(drop=True)


def _min_count(n: int, min_support: float) -> int:
    """Smallest integer count c with c / n >= min_support."""
    import math
    return max(1, math.ceil(min_support * n - 1e-9))


def generate_candidates(prev_level: list[tuple], prev_set: set[tuple], k: int) -> tuple[list[tuple], int, int]:
    """Self-join L_{k-1} with itself and prune with the Apriori property.

    ``prev_level`` holds sorted (k-1)-tuples.  Returns (candidates, n_joined, n_pruned).
    """
    prev_level = sorted(prev_level)
    joined: list[tuple] = []
    # group by common prefix of length k-2 so that the join is linear per group
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for itemset in prev_level:
        groups[itemset[:-1]].append(itemset)
    for prefix, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                joined.append(prefix + (members[i][-1], members[j][-1]))
    survivors = []
    for cand in joined:
        # prune: every (k-1)-subset must be frequent (downward closure)
        if all(sub in prev_set for sub in combinations(cand, k - 1)):
            survivors.append(cand)
    return survivors, len(joined), len(joined) - len(survivors)


def apriori(transactions: list[list[str]], min_support: float) -> AprioriResult:
    t0 = time.perf_counter()
    n = len(transactions)
    min_count = _min_count(n, min_support)
    tx = [tuple(sorted(set(t))) for t in transactions]

    # ---- scan 1: frequent 1-itemsets (L1)
    c1: dict[str, int] = defaultdict(int)
    for t in tx:
        for item in t:
            c1[item] += 1
    l1 = {(i,): c for i, c in c1.items() if c >= min_count}
    result = AprioriResult(n, min_support, min_count, {frozenset(k): v for k, v in l1.items()})
    result.levels.append(LevelStats(1, len(c1), 0, len(c1), len(l1)))
    frequent_items = {i for (i,) in l1}
    # transactions reduced to frequent items (items that are not frequent can
    # never be part of a frequent itemset - a consequence of downward closure)
    tx = [tuple(i for i in t if i in frequent_items) for t in tx]

    prev = list(l1.keys())
    k = 2
    while prev:
        cands, n_joined, n_pruned = generate_candidates(prev, set(prev), k)
        if not cands:
            result.levels.append(LevelStats(k, n_joined, n_pruned, 0, 0))
            break
        cand_set = set(cands)
        counts: dict[tuple, int] = defaultdict(int)
        # ---- database scan k: support counting.  For each transaction the
        # k-subsets are enumerated and looked up in the candidate hash set.
        for t in tx:
            if len(t) < k:
                continue
            for sub in combinations(t, k):
                if sub in cand_set:
                    counts[sub] += 1
        lk = {c: cnt for c, cnt in counts.items() if cnt >= min_count}
        result.levels.append(LevelStats(k, n_joined, n_pruned, len(cands), len(lk)))
        result.itemsets.update({frozenset(c): v for c, v in lk.items()})
        prev = list(lk.keys())
        k += 1
    result.runtime_s = time.perf_counter() - t0
    return result


def generate_rules(res: AprioriResult, min_confidence: float) -> pd.DataFrame:
    """For every frequent itemset l and every non-empty proper subset s,
    output s -> (l - s) if count(l) / count(s) >= min_confidence (Lecture 6, slide 9)."""
    n = res.n_transactions
    sup = res.itemsets
    rows = []
    for itemset, cnt in sup.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for ante in combinations(items, r):
                a = frozenset(ante)
                c = itemset - a
                conf = cnt / sup[a]
                if conf + 1e-12 < min_confidence:
                    continue
                s_xy, s_x, s_y = cnt / n, sup[a] / n, sup[c] / n
                lift = conf / s_y
                rows.append(dict(antecedent=", ".join(sorted(a)), consequent=", ".join(sorted(c)),
                                 antecedent_len=len(a), consequent_len=len(c), support_count=cnt,
                                 support=s_xy, confidence=conf, lift=lift, leverage=s_xy - s_x * s_y,
                                 conviction=float("inf") if conf >= 1 else (1 - s_y) / (1 - conf),
                                 coverage=s_x))
    cols = ["antecedent", "consequent", "antecedent_len", "consequent_len", "support_count", "support",
            "confidence", "lift", "leverage", "conviction", "coverage"]
    return pd.DataFrame(rows, columns=cols).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
