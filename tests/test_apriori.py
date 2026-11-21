"""Unit tests for the from-scratch Apriori (src/association_rules/apriori.py)."""
import math

import pandas as pd
import pytest

from src.association_rules.apriori import apriori, generate_candidates, generate_rules

# Textbook database used in Lecture 6 (Han & Kamber, AllElectronics): 9 transactions.
HAN_DB = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"], ["I2", "I3"],
          ["I1", "I3"], ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]


def test_support_counts_textbook():
    res = apriori(HAN_DB, min_support=2 / 9)
    s = res.itemsets
    assert s[frozenset(["I1"])] == 6 and s[frozenset(["I2"])] == 7 and s[frozenset(["I5"])] == 2
    assert s[frozenset(["I1", "I2"])] == 4
    # L3 of the textbook example
    l3 = {k for k in s if len(k) == 3}
    assert l3 == {frozenset(["I1", "I2", "I3"]), frozenset(["I1", "I2", "I5"])}
    assert not any(len(k) == 4 for k in s)


def test_rule_confidences_match_lecture6_slide9():
    res = apriori(HAN_DB, min_support=2 / 9)
    rules = generate_rules(res, 0.0)
    r = {(a, c): conf for a, c, conf in zip(rules.antecedent, rules.consequent, rules.confidence)}
    assert math.isclose(r[("I1, I2", "I5")], 2 / 4)
    assert math.isclose(r[("I1, I5", "I2")], 1.0)
    assert math.isclose(r[("I2, I5", "I1")], 1.0)
    assert math.isclose(r[("I1", "I2, I5")], 2 / 6)
    assert math.isclose(r[("I2", "I1, I5")], 2 / 7)
    assert math.isclose(r[("I5", "I1, I2")], 1.0)


def test_join_and_prune_lecture6_slide5():
    # L3 = {abc, abd, acd, ace, bcd}: join gives abcd and acde; acde pruned (ade not in L3)
    l3 = [tuple(x) for x in ("abc", "abd", "acd", "ace", "bcd")]
    cands, n_joined, n_pruned = generate_candidates(l3, set(l3), 4)
    assert set(cands) == {tuple("abcd")}
    assert n_joined == 2 and n_pruned == 1


def test_downward_closure_property():
    res = apriori(HAN_DB, min_support=2 / 9)
    for itemset in res.itemsets:
        for item in itemset:
            sub = itemset - {item}
            if sub:
                assert sub in res.itemsets and res.itemsets[sub] >= res.itemsets[itemset]


def test_rule_measures():
    res = apriori(HAN_DB, min_support=2 / 9)
    rules = generate_rules(res, 0.5)
    n = len(HAN_DB)
    for r in rules.itertuples():
        a = frozenset(r.antecedent.split(", "))
        c = frozenset(r.consequent.split(", "))
        sxy, sx, sy = res.itemsets[a | c] / n, res.itemsets[a] / n, res.itemsets[c] / n
        assert math.isclose(r.support, sxy)
        assert math.isclose(r.confidence, sxy / sx)
        assert math.isclose(r.lift, (sxy / sx) / sy)
        assert math.isclose(r.leverage, sxy - sx * sy)
        assert math.isclose(r.coverage, sx)
        assert r.confidence >= 0.5


def test_matches_mlxtend():
    from mlxtend.frequent_patterns import apriori as mx
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    X = pd.DataFrame(te.fit(HAN_DB).transform(HAN_DB), columns=te.columns_)
    ref = mx(X, min_support=2 / 9 - 1e-9, use_colnames=True)
    ref = {frozenset(s): round(v * 9) for s, v in zip(ref.itemsets, ref.support)}
    assert apriori(HAN_DB, 2 / 9).itemsets == ref


@pytest.mark.parametrize("ms", [0.3, 0.5, 0.9])
def test_higher_support_gives_subset(ms):
    lo = apriori(HAN_DB, 2 / 9).itemsets
    hi = apriori(HAN_DB, ms).itemsets
    assert set(hi) <= set(lo)
