"""
PHASE 8a - Decision Tree classification.

Problem definition
------------------
Target  : acclaimed = 1 if the film has >= 2 stars in Halliwell's Film Guide,
          0 if it is listed with 0-1 stars.  Only films that Halliwell rated
          are used (2 800 films); unrated films have no label and are NOT
          given a fabricated one.
Features: release-time attributes only (year, genres, colour process, studio
          group, director background/productivity, producer coding, cast
          size and gender composition, remake/book/series flags).
Excluded: Oscar wins/nominations, other critics' awards/ratings, 'was remade'
          (outcome information or known only later -> target leakage).

Procedure
---------
1. stratified 75/25 train/test split (seed 42) - stratification keeps the
   32 % positive rate in both parts
2. root-node comparison of Information Gain, Gain Ratio and Gini (own code)
3. 5-fold stratified cross-validated grid search on the training set over
   criterion {gini, entropy}, max_depth, min_samples_leaf, class_weight
4. cost-complexity (post-)pruning path: ccp_alpha chosen by CV
5. evaluation on the untouched test set: accuracy, precision, recall, F1,
   ROC-AUC, confusion matrix, classification report; baseline comparison
6. IF-THEN rule extraction, tree / ROC / importance / pruning plots

Run:  python -m src.classification.decision_tree
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from src.classification.split_criteria import entropy, gini, rank_attributes
from src.config import DT_DIR, PLOTS_DIR, PROCESSED_DIR, RANDOM_SEED, get_logger
from src.visualization.style import CAT, GREY, INK2, SEQ, apply_style, save

log = get_logger("decision_tree")
TARGET = "acclaimed"


def load_xy():
    d = pd.read_csv(PROCESSED_DIR / "dt_ml_matrix.csv")
    ids = d.pop("film_id")
    y = d.pop(TARGET).astype(int).to_numpy()
    return d, y, ids


def split(X, y, ids):
    return train_test_split(X, y, ids, test_size=0.25, stratify=y, random_state=RANDOM_SEED)


def extract_rules(tree: DecisionTreeClassifier, feature_names: list[str], Xtr: pd.DataFrame,
                  ytr: np.ndarray) -> pd.DataFrame:
    """Human-readable IF-THEN rules: one per leaf (root-to-leaf conjunction).

    The rule's class is the leaf prediction; 'training_positive_rate' and
    'samples' are the *unweighted* training films reaching the leaf (the
    tree is trained with balanced class weights, so stored leaf values are
    weighted and would overstate the minority class)."""
    t = tree.tree_
    leaf_of = tree.apply(Xtr)
    rules = []
    binary_cols = {c for c in feature_names if set(np.unique(Xtr[c])) <= {0.0, 1.0}}

    def recurse(node, conds):
        if t.children_left[node] == -1:
            mask = leaf_of == node
            n = int(mask.sum())
            pos = float(ytr[mask].mean()) if n else float("nan")
            cls = int(np.argmax(t.value[node][0]))
            rules.append(dict(rule=("IF " + " AND ".join(conds) if conds else "(root)") + f" THEN acclaimed = {cls}",
                              predicted_class=cls, training_positive_rate=pos,
                              rule_accuracy_on_train=pos if cls == 1 else 1 - pos, samples=n,
                              conditions=len(conds)))
            return
        f = feature_names[t.feature[node]]
        thr = t.threshold[node]
        if f in binary_cols:
            left, right = f"{f} = 0", f"{f} = 1"
        else:
            left, right = f"{f} <= {thr:.2f}", f"{f} > {thr:.2f}"
        recurse(t.children_left[node], conds + [left])
        recurse(t.children_right[node], conds + [right])

    recurse(0, [])
    return pd.DataFrame(rules).sort_values(["predicted_class", "rule_accuracy_on_train", "samples"],
                                           ascending=[False, False, False]).reset_index(drop=True)


def run() -> dict:
    log.info("PHASE 8a - decision tree started")
    apply_style()
    X, y, ids = load_xy()
    Xtr, Xte, ytr, yte, idtr, idte = split(X, y, ids)
    pd.DataFrame({"film_id": idtr, "split": "train"}).to_csv(DT_DIR / "train_ids.csv", index=False)
    pd.DataFrame({"film_id": idte, "split": "test"}).to_csv(DT_DIR / "test_ids.csv", index=False)

    # ---- (2) root-node attribute selection measures (own implementation)
    root = rank_attributes(Xtr, ytr)
    root.to_csv(DT_DIR / "root_split_measures.csv", index=False)
    root_stats = dict(entropy_root=entropy(ytr), gini_root=gini(ytr), positives_train=int(ytr.sum()),
                      n_train=len(ytr), n_test=len(yte), positives_test=int(yte.sum()))

    # ---- (3) CV grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = {"criterion": ["gini", "entropy"], "max_depth": [3, 4, 5, 6, 8, 10, None],
            "min_samples_leaf": [5, 10, 20, 40], "class_weight": [None, "balanced"]}
    gs = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_SEED), grid, scoring="roc_auc", cv=cv, n_jobs=1,
                      return_train_score=True)
    t0 = time.perf_counter()
    gs.fit(Xtr, ytr)
    grid_time = time.perf_counter() - t0
    cvres = pd.DataFrame(gs.cv_results_)
    cvres.to_csv(DT_DIR / "grid_search_cv_results.csv", index=False)
    crit_cmp = (cvres.groupby("param_criterion")["mean_test_score"].agg(["max", "mean"]).rename(
        columns={"max": "best_cv_roc_auc", "mean": "mean_cv_roc_auc"}))
    crit_cmp.to_csv(DT_DIR / "criterion_comparison.csv")
    best_params = gs.best_params_

    # ---- (4) cost-complexity pruning on top of the best pre-pruning settings
    base = DecisionTreeClassifier(random_state=RANDOM_SEED, **best_params)
    path = base.cost_complexity_pruning_path(Xtr, ytr)
    alphas = np.unique(np.round(path.ccp_alphas, 6))
    alphas = alphas[alphas >= 0][:-1] if len(alphas) > 1 else alphas
    if len(alphas) > 40:
        alphas = np.quantile(alphas, np.linspace(0, 1, 40))
    prune_rows = []
    for a in alphas:
        m = DecisionTreeClassifier(random_state=RANDOM_SEED, ccp_alpha=a, **best_params)
        cvs = cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        m.fit(Xtr, ytr)
        prune_rows.append(dict(ccp_alpha=a, cv_roc_auc=cvs, leaves=m.get_n_leaves(), depth=m.get_depth(),
                               train_acc=m.score(Xtr, ytr), test_acc=m.score(Xte, yte)))
    prune = pd.DataFrame(prune_rows)
    prune.to_csv(DT_DIR / "cost_complexity_pruning.csv", index=False)
    best_cv = prune["cv_roc_auc"].max()
    # one-standard-error style choice: simplest tree within 0.005 AUC of the best
    chosen = prune[prune["cv_roc_auc"] >= best_cv - 0.005].sort_values("ccp_alpha").iloc[-1]
    final_params = dict(best_params, ccp_alpha=float(chosen["ccp_alpha"]))

    # ---- (5) final model + evaluation on the test set
    clf = DecisionTreeClassifier(random_state=RANDOM_SEED, **final_params)
    t0 = time.perf_counter(); clf.fit(Xtr, ytr); fit_time = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = clf.predict(Xte); prob = clf.predict_proba(Xte)[:, 1]
    pred_time = time.perf_counter() - t0
    from sklearn.metrics import balanced_accuracy_score
    metrics = dict(accuracy=accuracy_score(yte, pred), balanced_accuracy=balanced_accuracy_score(yte, pred),
                   precision=precision_score(yte, pred),
                   recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, prob),
                   train_accuracy=clf.score(Xtr, ytr), leaves=int(clf.get_n_leaves()), depth=int(clf.get_depth()),
                   fit_time_s=fit_time, predict_time_s=pred_time)
    cv_final = cross_val_score(DecisionTreeClassifier(random_state=RANDOM_SEED, **final_params), Xtr, ytr, cv=cv,
                               scoring="accuracy")
    metrics["cv_accuracy_mean"], metrics["cv_accuracy_std"] = float(cv_final.mean()), float(cv_final.std())
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    metrics["baseline_majority_accuracy"] = float(dummy.score(Xte, yte))
    # criteria head-to-head with identical other settings
    head = {}
    for crit in ("gini", "entropy"):
        p = dict(final_params, criterion=crit)
        m = DecisionTreeClassifier(random_state=RANDOM_SEED, **p).fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        head[crit] = dict(test_accuracy=m.score(Xte, yte), test_f1=f1_score(yte, m.predict(Xte)),
                          test_roc_auc=roc_auc_score(yte, pr), leaves=int(m.get_n_leaves()))
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, target_names=["not acclaimed (0-1*)", "acclaimed (2+*)"], digits=3)
    (DT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]).to_csv(DT_DIR / "confusion_matrix.csv")
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(DT_DIR / "feature_importance.csv", header=["importance"])
    rules = extract_rules(clf, list(X.columns), Xtr, ytr)
    rules.to_csv(DT_DIR / "if_then_rules.csv", index=False)
    (DT_DIR / "tree_structure.txt").write_text(export_text(clf, feature_names=list(X.columns), show_weights=True),
                                               encoding="utf-8")

    figs = _plots(clf, X, yte, pred, prob, cm, imp, prune, chosen, root, cvres)
    summary = dict(target=TARGET, n_labelled=len(y), positive_rate=float(y.mean()), n_features=X.shape[1],
                   root=root_stats, best_cv_params=best_params, best_cv_roc_auc=float(gs.best_score_),
                   grid_search_time_s=grid_time, final_params=final_params, test_metrics=metrics,
                   criterion_head_to_head=head, criterion_cv=crit_cmp.reset_index().to_dict("records"),
                   top_root_attributes=root.head(8)[["attribute", "info_gain", "gain_ratio", "gini_reduction"]].to_dict("records"),
                   figures=figs, n_rules=len(rules))
    with open(DT_DIR / "decision_tree_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("PHASE 8a complete: %s", {k: summary[k] for k in ("final_params", "test_metrics")})
    return summary


def _plots(clf, X, yte, pred, prob, cm, imp, prune, chosen, root, cvres):
    out = []
    # tree (top levels for readability)
    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(clf, feature_names=list(X.columns), class_names=["not acclaimed", "acclaimed"], filled=True,
              max_depth=3, fontsize=8, impurity=True, proportion=False, rounded=True, ax=ax)
    ax.set_title(f"Decision tree (criterion = {clf.criterion}; top 4 levels of {clf.get_depth()}; "
                 f"{clf.get_n_leaves()} leaves)", fontsize=14)
    out.append(save(fig, PLOTS_DIR / "23_decision_tree_structure.png", dpi=130).name)
    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=SEQ)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / cm.sum():.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not acclaimed", "acclaimed"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not acclaimed", "acclaimed"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.grid(False)
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    out.append(save(fig, PLOTS_DIR / "24_decision_tree_confusion_matrix.png").name)
    # ROC
    fpr, tpr, _ = roc_curve(yte, prob)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color=CAT[0], label=f"decision tree (AUC = {roc_auc_score(yte, prob):.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="random classifier (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)"); ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (test set)"); ax.legend(loc="lower right")
    out.append(save(fig, PLOTS_DIR / "25_decision_tree_roc_curve.png").name)
    # feature importance
    top = imp[imp > 0].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top.index, top.values, color=CAT[0], height=0.65)
    for i, v in enumerate(top.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_title("Feature importance (total impurity decrease)")
    ax.set_xlabel("Normalised importance"); ax.set_ylabel("Feature")
    out.append(save(fig, PLOTS_DIR / "26_decision_tree_feature_importance.png").name)
    # pruning curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(prune["ccp_alpha"], prune["train_acc"], marker="o", ms=3, color=CAT[1], label="training accuracy")
    ax.plot(prune["ccp_alpha"], prune["test_acc"], marker="o", ms=3, color=CAT[0], label="test accuracy")
    ax.plot(prune["ccp_alpha"], prune["cv_roc_auc"], marker="o", ms=3, color=CAT[2], label="5-fold CV ROC-AUC")
    ax.axvline(chosen["ccp_alpha"], color=INK2, ls="--", lw=1)
    ax.text(chosen["ccp_alpha"], ax.get_ylim()[0] + 0.01, f" chosen alpha={chosen['ccp_alpha']:.4f}\n"
            f" ({int(chosen['leaves'])} leaves)", fontsize=9, color=INK2)
    ax.set_xlabel("Cost-complexity parameter alpha"); ax.set_ylabel("Score")
    ax.set_title("Cost-complexity pruning: accuracy vs tree complexity"); ax.legend()
    out.append(save(fig, PLOTS_DIR / "27_decision_tree_pruning_curve.png").name)
    # criteria comparison at the root
    r = root.head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, (m, lab, c) in zip(axes, [("info_gain", "Information gain", CAT[0]), ("gain_ratio", "Gain ratio", CAT[1]),
                                      ("gini_reduction", "Gini reduction", CAT[2])]):
        ax.barh(r["attribute"], r[m], color=c, height=0.65)
        ax.set_title(lab); ax.set_xlabel("Value at root node")
    axes[0].set_ylabel("Attribute (best binary split)")
    fig.suptitle("Root-node attribute selection measures (own implementation)", fontweight="bold")
    out.append(save(fig, PLOTS_DIR / "28_decision_tree_split_criteria.png").name)
    return out


if __name__ == "__main__":
    run()
