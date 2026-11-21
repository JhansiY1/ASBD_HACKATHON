"""
COFI-tree mining of frequent itemsets and extraction of CLOSED frequent
itemsets.

Algorithm source (not covered by the course slides - additional reference):
    M. El-Hajj and O. R. Zaiane, "COFI-tree Mining: A New Approach to Pattern
    Growth with Reduced Candidacy Generation", IEEE ICDM Workshop on Frequent
    Itemset Mining Implementations (FIMI'03), 2003.
Closed / maximal definitions (course slides, Lectures 10 and 11):
    X is CLOSED   iff no proper superset Y has support(Y) = support(X)
                  (equivalently f(X) = i(t(X)) = X, the A-Close closure operator)
    X is MAXIMAL  iff no proper superset is frequent.

Steps implemented
-----------------
1. Scan 1: count item supports; keep frequent items F, ordered by descending
   support (ties broken alphabetically) - the global FP-tree order.
2. Scan 2: build the FP-tree; each transaction is filtered to F, sorted by the
   global order and inserted as a path; shared prefixes share nodes; every
   item has a header chain (node-links) through its nodes.
3. For each frequent item A, from the LEAST frequent to the most frequent:
   a. Local frequency: follow A's node-links; for every A-node walk up to the
      root and add the A-node's count to every ancestor item.  Items whose
      co-occurrence count with A is below min_count are dropped ("locally
      infrequent" pruning - a major saving of COFI).
   b. Build the Co-Occurrence Frequent Item tree (COFI-tree) of A: root = A;
      each A-node's ancestor path (restricted to locally frequent items, kept
      in global order, most frequent next to the root) is inserted with the
      A-node's count.  Each COFI node stores (item, support count,
      participation count) and a node-link per item.
      Only items MORE frequent than A appear, so every pattern is generated in
      exactly one COFI-tree (the tree of its least frequent item) - no
      duplicates and no candidate explosion across trees.
   c. Mine the COFI-tree: items of the COFI header are visited from the least
      to the most frequent (i.e. deepest nodes first).  For every node N on
      the item's chain with  d = support(N) - participation(N) > 0  the path
      N -> ... -> root defines a *frequent-path-base* B with count d; the
      participation counter of every node on that path is increased by d.
      Because descendants are always processed before ancestors, the bases
      form a disjoint partition of the transactions containing A.
   d. Pattern generation: support({A} u S) = sum of counts of the bases that
      contain S, for every non-empty S that is a subset of some base.  Patterns
      with support >= min_count are frequent.
   e. The COFI-tree is discarded - only ONE small COFI-tree is in memory at a
      time (the memory advantage claimed by the authors).
4. Closed / maximal identification over the complete frequent set:
   X is closed iff no immediate superset X u {i} has the same support
   (sufficient because support is anti-monotone: an equal-support proper
   superset implies an equal-support immediate superset).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations


# =========================================================================
# FP-tree
# =========================================================================
class FPNode:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict = {}
        self.link = None


class FPTree:
    def __init__(self, transactions, min_count):
        counts = defaultdict(int)
        for t in transactions:
            for i in set(t):
                counts[i] += 1
        self.support = {i: c for i, c in counts.items() if c >= min_count}
        # global order: descending support, alphabetical tie-break
        self.order = sorted(self.support, key=lambda i: (-self.support[i], i))
        self.rank = {i: r for r, i in enumerate(self.order)}
        self.root = FPNode(None, None)
        self.head: dict = {}
        self.tail: dict = {}
        self.n_nodes = 0
        for t in transactions:
            items = sorted({i for i in t if i in self.support}, key=self.rank.get)
            self._insert(items)

    def _insert(self, items):
        node = self.root
        for i in items:
            child = node.children.get(i)
            if child is None:
                child = FPNode(i, node)
                node.children[i] = child
                self.n_nodes += 1
                if i in self.tail:
                    self.tail[i].link = child
                else:
                    self.head[i] = child
                self.tail[i] = child
            child.count += 1
            node = child

    def chain(self, item):
        n = self.head.get(item)
        while n is not None:
            yield n
            n = n.link


# =========================================================================
# COFI-tree
# =========================================================================
class COFINode:
    __slots__ = ("item", "count", "participation", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item = item
        self.count = 0
        self.participation = 0
        self.parent = parent
        self.children: dict = {}
        self.link = None


@dataclass
class COFITreeStats:
    item: str
    support: int
    local_items: int
    pruned_local_items: int
    nodes: int
    path_bases: int
    frequent_patterns: int


class COFITree:
    """Co-occurrence frequent-item tree of one item A."""

    def __init__(self, item, fp: FPTree, min_count):
        self.item = item
        self.root = COFINode(item, None)
        self.head: dict = {}
        self.tail: dict = {}
        self.n_nodes = 0
        a_nodes = list(fp.chain(item))
        self.root.count = sum(n.count for n in a_nodes)
        # (a) local frequencies of the more frequent items co-occurring with A
        local = defaultdict(int)
        for n in a_nodes:
            p = n.parent
            while p.item is not None:
                local[p.item] += n.count
                p = p.parent
        self.local_support = {i: c for i, c in local.items() if c >= min_count}
        self.pruned_local = len(local) - len(self.local_support)
        # header order: from least to most frequent (global order reversed)
        self.header_order = sorted(self.local_support, key=lambda i: -fp.rank[i])
        # (b) insert every A-node's prefix path (most frequent item next to the root)
        for n in a_nodes:
            path = []
            p = n.parent
            while p.item is not None:
                if p.item in self.local_support:
                    path.append(p.item)
                p = p.parent
            path.reverse()          # root-side (most frequent) first
            if path:
                self._insert(path, n.count)

    def _insert(self, path, count):
        node = self.root
        for i in path:
            child = node.children.get(i)
            if child is None:
                child = COFINode(i, node)
                node.children[i] = child
                self.n_nodes += 1
                if i in self.tail:
                    self.tail[i].link = child
                else:
                    self.head[i] = child
                self.tail[i] = child
            child.count += count
            node = child

    def mine(self, min_count) -> tuple[dict, int]:
        """(c)+(d): frequent-path-bases with participation counts, then
        pattern generation.  Returns ({pattern(frozenset incl. A): support}, #bases)."""
        bases: dict[frozenset, int] = defaultdict(int)
        for b_item in self.header_order:
            n = self.head.get(b_item)
            while n is not None:
                d = n.count - n.participation
                if d > 0:
                    path, p = [], n
                    while p is not self.root:
                        path.append(p.item)
                        p.participation += d
                        p = p.parent
                    bases[frozenset(path)] += d
                n = n.link
        support: dict[frozenset, int] = defaultdict(int)
        for base, cnt in bases.items():
            items = sorted(base)
            for r in range(1, len(items) + 1):
                for sub in combinations(items, r):
                    support[frozenset(sub)] += cnt
        a = self.item
        patterns = {frozenset([a]): self.root.count}
        for s, c in support.items():
            if c >= min_count:
                patterns[s | {a}] = c
        return patterns, len(bases)


@dataclass
class COFIResult:
    n_transactions: int
    min_support: float
    min_count: int
    frequent: dict[frozenset, int]
    closed: dict[frozenset, int] = field(default_factory=dict)
    maximal: dict[frozenset, int] = field(default_factory=dict)
    tree_stats: list[COFITreeStats] = field(default_factory=list)
    fp_tree_nodes: int = 0
    max_cofi_nodes: int = 0
    time_fp_tree_s: float = 0.0
    time_cofi_mining_s: float = 0.0
    time_closure_s: float = 0.0


def min_count_for(n, min_support):
    return max(1, math.ceil(min_support * n - 1e-9))


def cofi_mine(transactions, min_support=None, min_count=None) -> COFIResult:
    """Run COFI mining and closed/maximal identification."""
    n = len(transactions)
    if min_count is None:
        min_count = min_count_for(n, min_support)
    t0 = time.perf_counter()
    fp = FPTree(transactions, min_count)
    t1 = time.perf_counter()
    frequent: dict[frozenset, int] = {}
    stats = []
    max_nodes = 0
    for item in reversed(fp.order):                       # least frequent first
        ct = COFITree(item, fp, min_count)
        pats, nb = ct.mine(min_count)
        frequent.update(pats)
        max_nodes = max(max_nodes, ct.n_nodes)
        stats.append(COFITreeStats(item, ct.root.count, len(ct.local_support), ct.pruned_local,
                                   ct.n_nodes, nb, len(pats)))
        del ct                                            # discard the COFI-tree
    t2 = time.perf_counter()
    closed, maximal = closed_and_maximal(frequent, fp.order)
    t3 = time.perf_counter()
    return COFIResult(n, min_support if min_support is not None else min_count / n, min_count, frequent, closed,
                      maximal, stats, fp.n_nodes, max_nodes, t1 - t0, t2 - t1, t3 - t2)


def closed_and_maximal(frequent: dict, items: list) -> tuple[dict, dict]:
    """Closed: no immediate superset with equal support.  Maximal: no frequent immediate superset."""
    closed, maximal = {}, {}
    for x, s in frequent.items():
        is_closed, is_max = True, True
        for i in items:
            if i in x:
                continue
            sup = frequent.get(x | {i})
            if sup is not None:
                is_max = False
                if sup == s:
                    is_closed = False
                    break
        if is_closed:
            closed[x] = s
        if is_max:
            maximal[x] = s
    return closed, maximal


# =========================================================================
# Independent validation helpers
# =========================================================================
def brute_force_frequent(transactions, min_count) -> dict[frozenset, int]:
    """Exhaustive enumeration of every subset of every transaction (exact
    reference for small datasets)."""
    counts: dict[frozenset, int] = defaultdict(int)
    for t in transactions:
        items = sorted(set(t))
        for r in range(1, len(items) + 1):
            for sub in combinations(items, r):
                counts[frozenset(sub)] += 1
    return {k: v for k, v in counts.items() if v >= min_count}


def closure_operator(x: frozenset, tidsets: dict) -> frozenset:
    """A-Close closure f(X) = i(t(X)) (Lecture 11)."""
    tids = set.intersection(*(tidsets[i] for i in x))
    return frozenset(i for i, ts in tidsets.items() if tids <= ts)


def tidsets_of(transactions) -> dict:
    ts = defaultdict(set)
    for tid, t in enumerate(transactions):
        for i in t:
            ts[i].add(tid)
    return dict(ts)
