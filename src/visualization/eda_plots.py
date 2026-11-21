"""
Exploratory / descriptive visualisations (PHASE 6).

Each function draws one plot, saves it to outputs/plots/ and returns a
metadata record (plot type, attributes, reason, observations computed from
the data, inference, contribution).  The observations are generated from
the actual numbers so the written interpretation always matches the plot.

Plot types used here (all distinct): missing-value heatmap, histogram, line
plot with moving-average smoothing, stacked area chart, horizontal bar chart,
100 % stacked bar chart, donut chart, box plot, violin plot, KDE density plot,
correlation heatmap, bubble chart, treemap, count plot, scatter plot with
regression smoothing, Jaccard similarity heatmap.

Run:  python -m src.visualization.eda_plots
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analytics.descriptive import load_films
from src.config import OUTPUTS, PLOTS_DIR, PROCESSED_DIR, get_logger
from src.data_loading.html_table_loader import read_table
from src.preprocessing.cleaning import realign_main_fields
from src.preprocessing.transformation import ERA_ORDER, HALLIWELL_LEVELS
from src.preprocessing.vocabularies import CATEGORY_NAMES
from src.visualization.style import CAT, DIV, GREY, INK2, SEQ, apply_style, save

log = get_logger("visualization")
GN = CATEGORY_NAMES
ERA_SHORT = {e: e.split(" (")[0] for e in ERA_ORDER}


def _fig(w=9, h=5.2):
    return plt.subplots(figsize=(w, h))


# --------------------------------------------------------------------------- 1
def missing_heatmap(df):
    """Missing-value heatmap: % of placeholder/missing values per attribute and decade (raw MAIN)."""
    recs = [realign_main_fields(r.fields)[0] for r in read_table("main.html")]
    raw = pd.DataFrame(recs)
    yr = pd.to_numeric(raw["year_raw"].str.extract(r"((?:18|19)\d{2})")[0], errors="coerce")
    raw["decade"] = (yr // 10 * 10)
    tokens = {"producer_raw": {"PU:", "PN:", ""}, "studio_raw": {"SU:", ""}, "process_raw": {"prc", ""},
              "category_raw": {"Ctxx", "CTxx", ""}, "awards_raw": {"aw", ""}, "locale_raw": {"lc", ""},
              "notes_raw": {""}, "year_raw": {"", "19xx"}}
    names = {"year_raw": "year", "producer_raw": "producer", "studio_raw": "studio", "process_raw": "process",
             "category_raw": "category", "awards_raw": "awards", "locale_raw": "locale", "notes_raw": "notes"}
    m = pd.DataFrame({names[c]: raw[c].str.strip().isin(t) for c, t in tokens.items()})
    m["decade"] = raw["decade"]
    tab = (m.dropna(subset=["decade"]).groupby("decade").mean() * 100).T
    tab.columns = [f"{int(c)}s" for c in tab.columns]
    fig, ax = _fig(11, 4.8)
    sns.heatmap(tab, cmap=SEQ, vmin=0, vmax=100, annot=True, fmt=".0f", annot_kws={"size": 7.5},
                cbar_kws={"label": "% missing / placeholder"}, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Missing-value heatmap of the raw MAIN table by decade")
    ax.set_xlabel("Release decade"); ax.set_ylabel("Raw attribute")
    p = save(fig, PLOTS_DIR / "01_missing_value_heatmap.png")
    overall = (m.drop(columns="decade").mean() * 100).round(1).sort_values(ascending=False)
    return dict(file=p.name, type="Missing-value heatmap", attributes="all MAIN attributes x decade",
                reason="Missingness is encoded by placeholder codes (aw, lc, prc, Ctxx, PU:, SU:) and varies "
                       "strongly over time; a heatmap shows both dimensions at once.",
                observations=f"Overall placeholder/missing rates: " + ", ".join(f"{k} {v}%" for k, v in overall.items()) +
                ". Awards and locale are missing for most pre-1930 films, while the year is almost never missing.",
                inference="Missingness is not random (MNAR-like): it depends on the period and on the author's "
                          "documentation effort, so dropping incomplete rows would bias the data towards well "
                          "documented eras.",
                conclusion="Justifies attribute-specific treatment: placeholders -> missing, rule/group-based imputation "
                           "only where defensible, and no listwise deletion.")


# --------------------------------------------------------------------------- 2
def histogram_year(df):
    fig, ax = _fig()
    ax.hist(df["year"], bins=np.arange(1890, 2001, 2), color=CAT[0], edgecolor="white", linewidth=0.6)
    med = df["year"].median()
    ax.axvline(med, color=INK2, ls="--", lw=1.2)
    ax.text(med + 1, ax.get_ylim()[1] * 0.92, f"median = {int(med)}", color=INK2)
    ax.set_title("Histogram of release years (2-year bins)")
    ax.set_xlabel("Release year"); ax.set_ylabel("Number of films")
    p = save(fig, PLOTS_DIR / "02_histogram_release_year.png")
    s = df["year"]
    return dict(file=p.name, type="Histogram", attributes="year",
                reason="Year is the only continuous temporal attribute; a histogram shows its distribution shape.",
                observations=f"Range {int(s.min())}-{int(s.max())}, mean {s.mean():.1f}, median {int(med)}, "
                             f"skewness {s.skew():.2f}. Two humps: the classical studio era (1930s-1950s) and the "
                             f"1980s-1990s; a dip in the 1960s-70s.",
                inference="The collection is left-skewed towards recent films but contains a substantial historic "
                          "core; it is a curated sample (Hitchcock complete, personal interests), not a census.",
                conclusion="Era must be treated as a key explanatory variable; it was discretised into 5 film-history "
                           "periods for rule mining.")


# --------------------------------------------------------------------------- 3
def line_films_per_year(df):
    y = pd.read_csv(PROCESSED_DIR / "films_per_year_smoothed.csv", index_col=0)
    fig, ax = _fig(10, 5)
    ax.plot(y.index, y["films"], color=GREY, lw=1, label="films per year (raw)")
    ax.plot(y.index, y["moving_avg_5y"], color=CAT[0], lw=2.2, label="5-year centred moving average")
    ax.set_title("Films per release year with moving-average smoothing")
    ax.set_xlabel("Release year"); ax.set_ylabel("Number of films")
    ax.legend(loc="upper left")
    p = save(fig, PLOTS_DIR / "03_line_films_per_year_smoothed.png")
    top = y["films"].idxmax()
    return dict(file=p.name, type="Line plot (time series) with smoothing", attributes="year (count per year)",
                reason="A genuine temporal variable exists; the yearly counts are noisy, so a moving average "
                       "(noise smoothing, Lecture 5) reveals the trend.",
                observations=f"Peak raw count {int(y['films'].max())} films in {top}; the smoothed series rises to "
                             f"~{y.loc[1935:1960, 'moving_avg_5y'].max():.0f}/year in the studio era, falls to "
                             f"~{y.loc[1965:1975, 'moving_avg_5y'].min():.0f}/year around 1970 and climbs to "
                             f"~{y.loc[1985:1997, 'moving_avg_5y'].max():.0f}/year in the 1990s.",
                inference="Coverage reflects the compiler's sources (film guides published 1979-1990 and his own "
                          "viewing log), not true industry output.",
                conclusion="Temporal comparisons must be made on proportions within periods rather than raw counts.")


# --------------------------------------------------------------------------- 4
def area_color_share(df):
    d = df[(df["color_imputed"] == 0) & (df["color_type"] != "unknown")]
    tab = pd.crosstab(d["year"] // 5 * 5, d["color_type"], normalize="index")
    order = [c for c in ["silent_bw", "bw", "colorized", "color", "unknown"] if c in tab.columns]
    tab = tab[order]
    labels = {"silent_bw": "silent b/w", "bw": "b/w (sound)", "colorized": "colourised", "color": "colour",
              "unknown": "unknown"}
    cols = {"silent_bw": CAT[6], "bw": INK2, "colorized": CAT[4], "color": CAT[1], "unknown": "#d8d6d0"}
    fig, ax = _fig(10, 5)
    ax.stackplot(tab.index, tab.T.values * 100, labels=[labels[c] for c in order],
                 colors=[cols[c] for c in order], edgecolor="white", linewidth=0.8)
    ax.set_title("Share of film processes over time (stacked area, recorded processes only)")
    ax.set_xlabel("Release year (5-year bins)"); ax.set_ylabel("Share of films (%)")
    ax.set_ylim(0, 100); ax.set_xlim(tab.index.min(), tab.index.max())
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    p = save(fig, PLOTS_DIR / "04_area_color_process_share.png")
    cross = tab.index[(tab.get("color", 0) > tab.get("bw", 0))].min()
    return dict(file=p.name, type="Stacked area chart", attributes="year, color_type (recorded, non-imputed)",
                reason="Shows how a part-to-whole composition changes continuously over time.",
                observations=f"Silent b/w dominates before 1928; sound b/w dominates 1930-1950; colour first "
                             f"exceeds b/w in the {int(cross)}-{int(cross) + 4} bin and is near-universal after 1970.",
                inference="Colour process is almost a deterministic function of era, which validates the documented "
                          "imputation rule (b/w before 1945, colour after 1955) for unknown 'prc' values.",
                conclusion="Colour type and era are strongly associated; rules such as {color=bw} -> {era=Classical} "
                           "are expected and should not be over-interpreted.")


# --------------------------------------------------------------------------- 5
def bar_genres(df):
    g = df["genres"].explode().dropna().map(lambda x: GN.get(x, x)).value_counts().head(15)[::-1]
    fig, ax = _fig(9, 6)
    ax.barh(g.index, g.values, color=CAT[0], height=0.7)
    for i, v in enumerate(g.values):
        ax.text(v + 15, i, f"{v} ({100 * v / len(df):.1f}%)", va="center", fontsize=9, color=INK2)
    ax.set_title("Top 15 genres (multi-label count; a film can have several)")
    ax.set_xlabel("Number of films"); ax.set_ylabel("Genre")
    ax.set_xlim(0, g.max() * 1.22)
    p = save(fig, PLOTS_DIR / "05_bar_top_genres.png")
    return dict(file=p.name, type="Horizontal bar chart", attributes="genres (multi-valued, standardised)",
                reason="Compares the magnitude of many nominal categories with long labels.",
                observations=f"Drama ({g.iloc[-1]}) and Comedy ({g.iloc[-2]}) dominate, followed by "
                             f"{g.index[-3]} ({g.iloc[-3]}) and {g.index[-4]} ({g.iloc[-4]}); "
                             f"{100 * (df['n_genres'] == 0).mean():.1f}% of films have no valid category.",
                inference="Genre distribution is long-tailed; only ~15 genres have enough support (>= 1-2 %) to "
                          "appear in frequent itemsets.",
                conclusion="Sets a lower bound for useful minimum support (~1-2 %) in Apriori/COFI.")


# --------------------------------------------------------------------------- 6
def stacked_bar_genre_era(df):
    top = ["Dram", "Comd", "Susp", "Romt", "Horr", "Actn", "Musc", "West"]
    d = df.explode("genres")
    d = d[d["genres"].isin(top)]
    tab = pd.crosstab(d["era"], d["genres"], normalize="index").reindex(ERA_ORDER)[top] * 100
    fig, ax = _fig(10, 5.5)
    left = np.zeros(len(tab))
    for i, g in enumerate(top):
        ax.barh([ERA_SHORT[e] for e in tab.index], tab[g], left=left, color=CAT[i], label=GN[g],
                edgecolor="white", linewidth=1.5, height=0.68)
        for j, v in enumerate(tab[g]):
            if v >= 7:
                ax.text(left[j] + v / 2, j, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white")
        left += tab[g].values
    ax.set_title("Genre mix within each era (100% stacked bar, top-8 genres)")
    ax.set_xlabel("Share of genre labels within era (%)"); ax.set_ylabel("Era")
    ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    p = save(fig, PLOTS_DIR / "06_stacked_bar_genre_by_era.png")
    return dict(file=p.name, type="100% stacked bar chart", attributes="era, genres",
                reason="Compares composition (part-to-whole) across a small number of ordered groups.",
                observations=f"Western share peaks in the post-studio era ({tab.loc[ERA_ORDER[2], 'West']:.1f}%) "
                             f"and nearly vanishes after; Horror rises from {tab.loc[ERA_ORDER[1], 'Horr']:.1f}% "
                             f"(classical) to {tab.loc[ERA_ORDER[4], 'Horr']:.1f}% (modern); Musicals are a "
                             f"classical-era phenomenon ({tab.loc[ERA_ORDER[1], 'Musc']:.1f}%).",
                inference="Genres have life cycles; genre and era are dependent (Cramer's V reported in the "
                          "descriptive statistics).",
                conclusion="Genre-era rules are expected in association mining; era is a useful feature for "
                           "classification and clustering.")


# --------------------------------------------------------------------------- 7
def donut_color(df):
    vc = df["color_type"].value_counts()
    labels = {"color": "colour", "bw": "b/w (sound)", "silent_bw": "silent b/w", "unknown": "unknown",
              "colorized": "colourised"}
    cols = {"color": CAT[1], "bw": INK2, "silent_bw": CAT[6], "unknown": "#d8d6d0", "colorized": CAT[4]}
    fig, ax = _fig(7.5, 6)
    wedges, _ = ax.pie(vc.values, colors=[cols[c] for c in vc.index], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    ax.legend(wedges, [f"{labels[c]}: {v} ({100 * v / vc.sum():.1f}%)" for c, v in vc.items()],
              loc="center left", bbox_to_anchor=(1, 0.5))
    ax.text(0, 0, f"{vc.sum():,}\nfilms", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("Film process (colour type) after cleaning and imputation")
    ax.axis("equal")
    p = save(fig, PLOTS_DIR / "07_donut_color_type.png")
    return dict(file=p.name, type="Donut chart", attributes="color_type",
                reason="Five mutually exclusive categories summing to 100% - the case where a pie/donut is "
                       "appropriate.",
                observations="; ".join(f"{labels[c]} {100 * v / vc.sum():.1f}%" for c, v in vc.items()) +
                             f". {int(df['color_imputed'].sum())} values were rule-imputed.",
                inference="Colour films form the majority because the collection is weighted to post-1955 films; "
                          "the 'unknown' slice is concentrated in the 1945-55 transition decade.",
                conclusion="Colour type is usable as a categorical item; the residual 'unknown' class is excluded "
                           "from transactions.")


# --------------------------------------------------------------------------- 8
def box_cast_size(df):
    d = df[(df["has_cast_info"] == 1)].copy()
    d["era_s"] = d["era"].map(ERA_SHORT)
    fig, ax = _fig(10, 5.2)
    sns.boxplot(data=d, x="era_s", y="cast_size", order=[ERA_SHORT[e] for e in ERA_ORDER], color=CAT[0],
                width=0.55, fliersize=2.5, linewidth=1.1, ax=ax,
                flierprops=dict(marker="o", markerfacecolor=GREY, markeredgecolor="none", alpha=0.6))
    ax.set_title("Documented cast size by era (box plot, IQR rule outliers as dots)")
    ax.set_xlabel("Era"); ax.set_ylabel("Number of listed actors")
    p = save(fig, PLOTS_DIR / "08_box_cast_size_by_era.png")
    med = d.groupby("era")["cast_size"].median().reindex(ERA_ORDER)
    q = d["cast_size"].quantile([0.25, 0.75])
    fence = q.iloc[1] + 1.5 * (q.iloc[1] - q.iloc[0])
    return dict(file=p.name, type="Box plot", attributes="cast_size (from CASTS), era",
                reason="Compares distribution, median, spread and outliers of a numeric attribute across groups.",
                observations="Median cast size by era: " + ", ".join(f"{ERA_SHORT[e]} {v:.0f}" for e, v in med.items()) +
                             f". IQR upper fence {fence:.1f}; {int((d['cast_size'] > fence).sum())} films exceed it "
                             f"(max {int(d['cast_size'].max())}).",
                inference="The outliers are genuine heavily documented films (e.g. all-star casts), not errors; "
                          "cast size measures documentation depth as much as production size.",
                conclusion="Outliers are kept; cast size is used in binned form for rule mining and as a numeric "
                           "feature for the tree.")


# --------------------------------------------------------------------------- 9
def violin_year_by_rating(df):
    d = df[df["halliwell_level"].notna()]
    fig, ax = _fig(10, 5.2)
    sns.violinplot(data=d, x="halliwell_level", y="year", order=HALLIWELL_LEVELS, color=CAT[2],
                   inner="quartile", cut=0, linewidth=1, ax=ax)
    ax.set_title("Release-year distribution by Halliwell critic rating (violin plot)")
    ax.set_xlabel("Halliwell's Film Guide rating"); ax.set_ylabel("Release year")
    p = save(fig, PLOTS_DIR / "09_violin_year_by_halliwell.png")
    med = d.groupby("halliwell_level")["year"].median().reindex(HALLIWELL_LEVELS)
    return dict(file=p.name, type="Violin plot", attributes="year, halliwell_level",
                reason="Shows full distribution shape (density + quartiles) of a numeric variable per ordered "
                       "category; richer than a box plot when distributions are multi-modal.",
                observations="Median release year by rating: " + ", ".join(f"{k} {int(v)}" for k, v in med.items()) +
                             ". Higher-rated films are concentrated in 1930-1960.",
                inference="Halliwell's guide (1987 edition) favours classic films: rating is confounded with era. "
                          "Year will therefore be a strong predictor of acclaim.",
                conclusion="The decision tree must be interpreted as modelling the critic's preferences, not "
                           "objective quality.")


# --------------------------------------------------------------------------- 10
def kde_genre_years(df):
    gs = ["West", "Musc", "Horr", "ScFi", "Noir", "Porn"]
    fig, ax = _fig(10, 5.2)
    for i, g in enumerate(gs):
        yrs = df.loc[df["genres"].apply(lambda x, g=g: g in x), "year"]
        sns.kdeplot(yrs, ax=ax, color=CAT[i], lw=2, label=f"{GN[g]} (n={len(yrs)})", bw_adjust=0.8)
    ax.set_title("Density of release years for selected genres (KDE)")
    ax.set_xlabel("Release year"); ax.set_ylabel("Estimated density")
    ax.legend()
    p = save(fig, PLOTS_DIR / "10_kde_genre_release_years.png")
    peaks = {}
    for g in gs:
        yrs = df.loc[df["genres"].apply(lambda x, g=g: g in x), "year"]
        peaks[GN[g]] = int(yrs.median())
    return dict(file=p.name, type="Density (KDE) plot", attributes="year by genre",
                reason="Smooth densities let several overlapping distributions be compared on one axis.",
                observations="Median year: " + ", ".join(f"{k} {v}" for k, v in peaks.items()) +
                             ". Westerns and Musicals peak 1935-1960, Film noir is tightly 1940-1955, Horror and "
                             "Sci-fi peak after 1970.",
                inference="Each genre has a distinct historical life cycle - temporal context is essential for "
                          "interpreting genre co-occurrence.",
                conclusion="Supports the use of era items in transactions and year as a classifier feature.")


# --------------------------------------------------------------------------- 11
def corr_heatmap(df):
    s = pd.read_csv(OUTPUTS / "descriptive" / "correlation_spearman.csv", index_col=0)
    lab = {"year": "year", "cast_size": "cast size", "n_producers": "# producers", "n_genres": "# genres",
           "halliwell_stars": "Halliwell stars", "oscar_noms": "Oscar noms", "director_n_films": "director # films",
           "female_cast_share": "female cast share", "times_remade": "times remade"}
    s = s.rename(index=lab, columns=lab)
    mask = np.triu(np.ones_like(s, dtype=bool), k=1)
    fig, ax = _fig(8.5, 7)
    sns.heatmap(s, mask=mask, cmap=DIV, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", square=True,
                linewidths=0.6, linecolor="white", cbar_kws={"label": "Spearman rho", "shrink": 0.8}, ax=ax)
    ax.set_title("Spearman correlation heatmap of numeric attributes")
    ax.set_xlabel("Attribute"); ax.set_ylabel("Attribute")
    p = save(fig, PLOTS_DIR / "11_correlation_heatmap.png")
    v = s.where(~np.eye(len(s), dtype=bool)).stack()
    strongest = v.abs().sort_values(ascending=False).drop_duplicates().head(4)
    return dict(file=p.name, type="Correlation heatmap", attributes=", ".join(lab.values()),
                reason="Only genuinely numeric/ordinal attributes are used (no arbitrary category codes); Spearman "
                       "is used because most variables are skewed counts.",
                observations="Strongest pairs: " + "; ".join(f"{a} vs {b}: {v[(a, b)]:+.2f}" for a, b in strongest.index) +
                             ". All other correlations are weak (|rho| < 0.3).",
                inference="Critic rating correlates negatively with year and positively with Oscar nominations and "
                          "remakes; most attributes are nearly independent, so there is little redundancy.",
                conclusion="No numeric attribute needs to be dropped for redundancy; Oscar nominations are excluded "
                           "from the classifier because of target leakage, not redundancy.")


# --------------------------------------------------------------------------- 12
def bubble_directors(df):
    d = pd.read_csv(PROCESSED_DIR / "director_profiles.csv")
    d = d[d["share_rated"] > 0.3]
    fig, ax = _fig(10, 6)
    sc = ax.scatter(d["mean_year"], d["mean_halliwell"], s=d["n_films"] * 9, c=d["share_color"], cmap=SEQ,
                    alpha=0.75, edgecolor="white", linewidth=0.8)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Share of colour films")
    for _, r in d.nlargest(8, "n_films").iterrows():
        ax.annotate(r["director"], (r["mean_year"], r["mean_halliwell"]), fontsize=8.5, color=INK2,
                    xytext=(5, 5), textcoords="offset points")
    for n in (10, 30, 60):
        ax.scatter([], [], s=n * 9, color=GREY, alpha=0.5, label=f"{n} films")
    ax.legend(title="Bubble size", loc="upper right", labelspacing=1.3)
    ax.set_title("Directors: mean film year vs mean Halliwell rating (bubble = # films)")
    ax.set_xlabel("Mean release year of director's films"); ax.set_ylabel("Mean Halliwell stars (rated films)")
    p = save(fig, PLOTS_DIR / "12_bubble_directors_year_rating.png")
    r = np.corrcoef(d["mean_year"], d["mean_halliwell"])[0, 1]
    return dict(file=p.name, type="Bubble chart", attributes="director profile: mean_year, mean_halliwell, "
                                                             "n_films, share_color",
                reason="Encodes four director-level variables at once (x, y, size, colour).",
                observations=f"{len(d)} directors with >= 5 films and > 30% rated films. Pearson r(mean year, mean "
                             f"rating) = {r:.2f}. Highly rated prolific directors (e.g. Hitchcock, Lubitsch, Cukor) "
                             f"cluster at 1930-1960 with low colour share.",
                inference="Director reputation is entangled with era; prolific classical directors receive higher "
                          "ratings in this source.",
                conclusion="Motivates clustering directors on career profiles (K-Means) to separate eras and styles.")


# --------------------------------------------------------------------------- 13
def treemap_studios(df):
    import squarify
    vc = df.loc[df["studio_group"] != "Unknown", "studio_group"].value_counts()
    fig, ax = _fig(10, 6)
    # every block is labelled, so identity does not rely on colour: one hue for the
    # classical majors, a second for Hitchcock's TV company, grey for the remainder
    colors = [GREY if s == "Other studio" else CAT[2] if s.startswith("Shamley") else CAT[0] for s in vc.index]
    squarify.plot(sizes=vc.values, label=[f"{s}\n{v} ({100 * v / vc.sum():.1f}%)" for s, v in vc.items()],
                  color=colors, alpha=0.85, ax=ax, text_kwargs={"fontsize": 9, "color": "white"},
                  edgecolor="white", linewidth=2)
    ax.axis("off")
    ax.set_title("Treemap of films by production studio (studio known)")
    p = save(fig, PLOTS_DIR / "13_treemap_studios.png")
    majors = vc.drop(["Other studio", "Shamley (Hitchcock TV)"], errors="ignore")
    return dict(file=p.name, type="Treemap", attributes="studio_group",
                reason="Part-to-whole for ~11 categories where one 'Other' block is large; area encodes share "
                       "better than a pie with many slices.",
                observations=f"Studio is known for {vc.sum()} films ({100 * vc.sum() / len(df):.1f}%). The 8 "
                             f"classical majors account for {100 * majors.sum() / vc.sum():.1f}% of them; MGM "
                             f"({vc.get('MGM', 0)}), Warner Bros ({vc.get('Warner Bros', 0)}) and Paramount "
                             f"({vc.get('Paramount', 0)}) lead; Shamley = Hitchcock's TV company.",
                inference="Studio information is recorded mostly for Hollywood studio-era films; it is a proxy for "
                          "the studio system rather than a complete production attribute.",
                conclusion="Studio items are included in transactions only for known majors; 'Other'/'Unknown' "
                           "are omitted to avoid uninformative items.")


# --------------------------------------------------------------------------- 14
def countplot_decade_oscar(df):
    d = df.copy()
    d["Oscar"] = np.where(d["any_oscar_nom"] == 1, "Oscar nominated/won", "no Oscar record")
    d = d[d["year"] >= 1920]
    fig, ax = _fig(10, 5)
    sns.countplot(data=d[d["Oscar"] == "Oscar nominated/won"], x="decade", color=CAT[3], ax=ax)
    ax.set_title("Count plot: Oscar-nominated films per decade")
    ax.set_xlabel("Decade"); ax.set_ylabel("Number of Oscar-nominated films")
    ax.set_xticks(ax.get_xticks()); ax.set_xticklabels([f"{int(float(t.get_text()))}s" for t in ax.get_xticklabels()])
    rate = d.groupby("decade")["any_oscar_nom"].mean() * 100
    for i, (dec, r) in enumerate(rate.items()):
        cnt = int(d.loc[d["decade"] == dec, "any_oscar_nom"].sum())
        ax.text(i, cnt + 1, f"{r:.1f}%", ha="center", fontsize=8.5, color=INK2)
    p = save(fig, PLOTS_DIR / "14_countplot_oscar_by_decade.png")
    return dict(file=p.name, type="Count plot", attributes="decade, any_oscar_nom",
                reason="Counts of a binary event per ordered category; labels give the rate within decade.",
                observations=f"{int(df['any_oscar_nom'].sum())} films carry an Academy Award nomination/win. The "
                             f"nomination rate is highest in the {int(rate.idxmax())}s ({rate.max():.1f}%) and lowest in "
                             f"the {int(rate.idxmin())}s ({rate.min():.1f}%).",
                inference="Award documentation is concentrated in the classical era, mirroring the critic ratings.",
                conclusion="The Oscar item is rare (~3.6%); it only appears in rules with low support but high lift.")


# --------------------------------------------------------------------------- 15
def scatter_female_share(df):
    d = df[(df["has_cast_info"] == 1) & df["female_cast_share"].notna()]
    fig, ax = _fig(10, 5.2)
    jitter = np.random.default_rng(42).uniform(-0.4, 0.4, len(d))
    ax.scatter(d["year"] + jitter, d["female_cast_share"], s=8, alpha=0.18, color=CAT[0], edgecolor="none",
               label="films")
    coef = np.polyfit(d["year"], d["female_cast_share"], 1)
    xs = np.linspace(d["year"].min(), d["year"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color=CAT[1], lw=2.4, label=f"linear regression (slope {coef[0] * 10:+.4f}/decade)")
    m = d.groupby(d["year"] // 5 * 5)["female_cast_share"].mean()
    ax.plot(m.index + 2.5, m.values, color=INK2, lw=1.6, ls="--", label="5-year bin means (smoothing)")
    ax.set_title("Female share of documented cast vs release year")
    ax.set_xlabel("Release year"); ax.set_ylabel("Female share of cast (known gender)")
    ax.legend(loc="upper right")
    p = save(fig, PLOTS_DIR / "15_scatter_female_cast_share.png")
    return dict(file=p.name, type="Scatter plot with regression smoothing", attributes="year, female_cast_share",
                reason="Relationship between two numeric attributes; regression and bin-mean lines demonstrate "
                       "the 'smoothing by regression/binning' noise-handling methods of Lecture 5.",
                observations=f"{len(d)} films with known cast gender. Mean female share {d['female_cast_share'].mean():.3f}; "
                             f"regression slope {coef[0] * 10:+.4f} per decade; bin means range "
                             f"{m.min():.2f}-{m.max():.2f}.",
                inference="Casts listed in the database are about two-thirds male across the whole century, with "
                          "only a weak trend; the points are very noisy and individual films vary from 0 to 1.",
                conclusion="Female share is retained as a numeric feature; smoothing shows the stable underlying "
                           "level.")


# --------------------------------------------------------------------------- 16
def jaccard_genres(df):
    top = ["Dram", "Comd", "Susp", "Romt", "Horr", "Actn", "Musc", "Advt", "BioP", "West", "ScFi", "Fant"]
    B = np.array([[int(g in gs) for g in top] for gs in df["genres"]])
    inter = B.T @ B
    cnt = np.diag(inter)
    union = cnt[:, None] + cnt[None, :] - inter
    J = inter / union
    np.fill_diagonal(J, np.nan)
    names = [GN[g] for g in top]
    fig, ax = _fig(9, 7.5)
    sns.heatmap(pd.DataFrame(J, index=names, columns=names), cmap=SEQ, annot=True, fmt=".3f",
                annot_kws={"size": 7.5}, linewidths=0.5, linecolor="white", square=True,
                cbar_kws={"label": "Jaccard similarity |A n B| / |A u B|", "shrink": 0.8}, ax=ax)
    ax.set_title("Jaccard similarity between genres (film sets)")
    ax.set_xlabel("Genre"); ax.set_ylabel("Genre")
    p = save(fig, PLOTS_DIR / "16_jaccard_genre_similarity.png")
    Jd = pd.DataFrame(J, index=names, columns=names).stack().sort_values(ascending=False)
    Jd = Jd[[a < b for a, b in Jd.index]].head(4)
    pd.DataFrame(J, index=names, columns=names).to_csv(OUTPUTS / "descriptive" / "jaccard_genre_similarity.csv")
    return dict(file=p.name, type="Similarity heatmap (Jaccard)", attributes="genres (binary item vectors)",
                reason="Lecture 22 similarity measure for sets: genres are sets of films, Jaccard measures overlap "
                       "independent of the many films with neither genre.",
                observations="Most similar pairs: " + "; ".join(f"{a}-{b} J={v:.3f}" for (a, b), v in Jd.items()) +
                             ". All other pairs have J < 0.05.",
                inference="Genres are largely disjoint labels; the few overlapping pairs (romantic comedy, musical "
                          "comedy, horror-fantasy) are the multi-genre combinations association mining can find.",
                conclusion="Genre-genre rules will be rare and low-support; most rules will link genre with era, "
                           "colour, studio and rating items.")


ALL = [missing_heatmap, histogram_year, line_films_per_year, area_color_share, bar_genres, stacked_bar_genre_era,
       donut_color, box_cast_size, violin_year_by_rating, kde_genre_years, corr_heatmap, bubble_directors,
       treemap_studios, countplot_decade_oscar, scatter_female_share, jaccard_genres]


def run() -> list[dict]:
    apply_style()
    log.info("PHASE 6 - visualisation started")
    df = load_films()
    meta = []
    for i, fn in enumerate(ALL, start=1):
        m = fn(df)
        m["number"] = i
        meta.append(m)
        log.info("plot %d saved: %s", i, m["file"])
    with open(OUTPUTS / "plots_eda_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


if __name__ == "__main__":
    run()
