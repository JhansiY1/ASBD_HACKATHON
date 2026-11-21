"""Unit tests for COFI-tree mining and closed/maximal itemset identification."""
import random

import pytest

from src.cofi.cofi import (FPTree, brute_force_frequent, closed_and_maximal, closure_operator, cofi_mine,
                           tidsets_of)

# Lecture 11 A-Close example (transactions reconstructed from the slide's i(generators) column)
LECT11 = [["A", "C"], ["B", "C", "E"], ["A", "B", "C", "E"], ["B", "E"], ["A", "B", "C", "E"]]


def test_lecture11_closed_sets():
    res = cofi_mine(LECT11, min_count=3)
    assert {frozenset(x) for x in res.closed} == {frozenset("AC"), frozenset("BE"), frozenset("BCE"), frozenset("C")}
    assert res.closed[frozenset("BE")] == 4 and res.closed[frozenset("BCE")] == 3


def test_lecture10_closed_vs_maximal():
    # DB {a1..a10}, {a1..a5} with min_sup = 1 (scaled-down version of the Lecture 10 example)
    big = [f"a{i}" for i in range(1, 11)]
    db = [big, big[:5]]
    res = cofi_mine(db, min_count=1)
    assert res.closed == {frozenset(big): 1, frozenset(big[:5]): 2}
    assert res.maximal == {frozenset(big): 1}
    assert len(res.frequent) == 2 ** 10 - 1


def test_fp_tree_counts():
    fp = FPTree(LECT11, 1)
    assert fp.support == {"A": 3, "B": 4, "C": 4, "E": 4}
    total_root_children = sum(c.count for c in fp.root.children.values())
    assert total_root_children == len(LECT11)


@pytest.mark.parametrize("seed", range(15))
def test_random_against_bruteforce(seed):
    rng = random.Random(seed)
    items = [chr(97 + i) for i in range(rng.randint(4, 8))]
    db = [rng.sample(items, rng.randint(1, len(items))) for _ in range(rng.randint(5, 30))]
    mc = rng.randint(1, 4)
    res = cofi_mine(db, min_count=mc)
    bf = brute_force_frequent(db, mc)
    assert res.frequent == bf
    ts = tidsets_of(db)
    assert res.closed == {x: s for x, s in bf.items() if closure_operator(x, ts) == x}


def test_closed_definition_holds():
    res = cofi_mine(LECT11, min_count=2)
    for x, s in res.closed.items():
        for y, t in res.frequent.items():
            if x < y:
                assert t < s                        # no equal-support proper superset
    for x in res.maximal:
        assert not any(x < y for y in res.frequent)


def test_closed_and_maximal_helper():
    freq = {frozenset("a"): 3, frozenset("b"): 2, frozenset("ab"): 2}
    closed, maximal = closed_and_maximal(freq, ["a", "b"])
    assert set(closed) == {frozenset("a"), frozenset("ab")}
    assert set(maximal) == {frozenset("ab")}
