"""
Shared helpers for the nanoparticle-synthesis notebooks.

Three groups of functions:
  1. parsing      - turn messy spreadsheet cells into numbers / clean strings
  2. evaluation   - leakage-free, paper-grouped, repeated cross-validation with baselines
  3. plotting     - small, consistent matplotlib style

Dataset-specific cleaning and feature engineering live in the notebooks themselves,
so that every decision is visible next to its explanation.
"""
from __future__ import annotations

import datetime as dt
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, r2_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

# --------------------------------------------------------------------------- #
# 1. Parsing
# --------------------------------------------------------------------------- #
# Colours: first three slots of a colour-blind-validated categorical palette + neutrals.
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#2b2b2a"


def fix_text(s):
    """Repair UTF-8 text that was mis-decoded as cp1252 (e.g. 'Â·' -> '·'); leave anything else alone."""
    if not isinstance(s, str):
        return s
    s = s.replace("_x000D_", " ").replace("\r", " ").replace("\n", " ")
    try:
        s = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return re.sub(r"\s+", " ", s).strip()


def parse_number(x):
    """
    Robust numeric parser for hand-entered cells.
      '0.005' -> 0.005     ' 12 ' -> 12      '5-30' / '5–30' -> 17.5 (range midpoint)
      '2:1' -> 2.0         datetime.time(2, 1) -> 2.0  (Excel turns '2:1' into a clock time)
      '300 rpm' -> 300     anything else -> NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, dt.time):                       # '2:1' read as 02:01:00
        return x.hour / x.minute if x.minute else np.nan
    if isinstance(x, (dt.datetime, pd.Timestamp)):
        return np.nan
    s = str(x).strip().replace("–", "-").replace("—", "-")
    if not s:
        return np.nan
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)(?::0+)?", s)   # ratio a:b (or a:b:00)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else np.nan
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)           # range a-b
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.match(r"^(\d+(?:\.\d+)?)\s*[a-zA-Z%]*$", s)                      # number + trailing unit
    return float(m.group(1)) if m else np.nan


def mad(x):
    x = np.asarray(x, float)
    return np.nanmedian(np.abs(x - np.nanmedian(x)))


# --------------------------------------------------------------------------- #
# 2. Baselines that need the raw table
# --------------------------------------------------------------------------- #
class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Predict the median target of the row's category (e.g. nanoparticle family); fall back to the global median."""

    def __init__(self, col="np_family"):
        self.col = col

    def fit(self, X, y, **kw):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = float(y.median())
        self.med_ = y.groupby(X[self.col]).median().to_dict()
        return self

    def predict(self, X):
        return X[self.col].map(self.med_).fillna(self.global_).to_numpy(float)


class GroupPriorClassifier(BaseEstimator, ClassifierMixin):
    """Predict the positive-class rate of the row's category; fall back to the global rate."""

    def __init__(self, col="np_family"):
        self.col = col

    def fit(self, X, y, **kw):
        y = pd.Series(np.asarray(y), index=X.index)
        self.classes_ = np.array([0, 1])
        self.global_ = float(y.mean())
        self.rate_ = y.groupby(X[self.col]).mean().to_dict()
        return self

    def predict_proba(self, X):
        p = X[self.col].map(self.rate_).fillna(self.global_).to_numpy(float)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# 3. Evaluation harness
# --------------------------------------------------------------------------- #
def _fit(est, X, y, groups):
    """GridSearchCV needs `groups` so its *inner* folds also keep papers together."""
    if isinstance(est, GridSearchCV):
        return est.fit(X, y, groups=groups)
    return est.fit(X, y)


def repeated_group_cv(estimator, X, y, groups, task, n_splits=5, n_repeats=5, seed=0):
    """
    Repeated paper-grouped K-fold. All rows of one paper are always in the same fold,
    so the model is always scored on papers it has never seen.

    Returns
      oof   : (n_repeats, n) out-of-fold predictions (value for regression, P(class 1) for classification)
      score : DataFrame, one row per repeat, with the metrics for that repeat
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    n = len(y)
    oof = np.full((n_repeats, n), np.nan)
    folds = np.zeros((n_repeats, n), dtype=int)
    for r in range(n_repeats):
        if task == "reg":
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            splits = splitter.split(X, y, groups)
        else:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            splits = splitter.split(X, y, groups)
        for k, (tr, te) in enumerate(splits):
            est = _fit(clone(estimator), X.iloc[tr], y[tr], groups[tr])
            oof[r, te] = est.predict(X.iloc[te]) if task == "reg" else est.predict_proba(X.iloc[te])[:, 1]
            folds[r, te] = k
    rows = [regression_metrics(y, oof[r]) if task == "reg" else classification_metrics(y, oof[r], folds[r])
            for r in range(n_repeats)]
    return oof, pd.DataFrame(rows)


def regression_metrics(y_log, p_log):
    """y, p are log10(size in nm). Report errors in log space and as 'fold error' in nm terms."""
    err = np.abs(p_log - y_log)
    return {
        "MAE_log10": err.mean(),
        "fold_error": 10 ** err.mean(),                       # typical multiplicative miss, e.g. 1.8 = 'off by 1.8x'
        "R2_log": r2_score(y_log, p_log),
        "median_abs_err_nm": np.median(np.abs(10 ** p_log - 10 ** y_log)),
        "within_2x": np.mean(err <= np.log10(2)),
    }


def classification_metrics(y, p, folds=None):
    """
    ROC-AUC, PR-AUC and balanced accuracy are computed *within each test fold* and averaged
    (pooling folds mixes differently-calibrated models and makes a constant predictor look worse than chance).
    Brier score is pooled. Folds with a single class are skipped for the ranking metrics.
    """
    y, p = np.asarray(y), np.asarray(p)
    folds = np.zeros(len(y), int) if folds is None else folds
    auc, ap, ba = [], [], []
    for k in np.unique(folds):
        m = folds == k
        if len(np.unique(y[m])) < 2:
            continue
        auc.append(roc_auc_score(y[m], p[m]))
        ap.append(average_precision_score(y[m], p[m]))
        ba.append(balanced_accuracy_score(y[m], (p[m] >= 0.5).astype(int)))
    return {"ROC_AUC": np.mean(auc), "PR_AUC": np.mean(ap), "Brier": brier_score_loss(y, p),
            "balanced_acc@0.5": np.mean(ba)}


def per_row_loss(y, oof, task):
    """Mean over repeats of a per-row loss, so that models can be compared row by row."""
    y = np.asarray(y)
    loss = np.abs(oof - y) if task == "reg" else (oof - y) ** 2          # abs log10 error  |  Brier
    return loss.mean(axis=0)


def paired_group_bootstrap(loss_a, loss_b, groups, n_boot=4000, seed=0):
    """
    Mean(loss_a) - mean(loss_b) with a 95% CI, resampling *papers* (not rows) with replacement.
    Negative = model A is better (lower loss) than B.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    d = np.asarray(loss_a) - np.asarray(loss_b)
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(ug), len(ug))
        boots[i] = sums[idx].sum() / cnts[idx].sum()
    return d.mean(), np.percentile(boots, [2.5, 97.5])


def summarize(results, metric):
    """results: {model name: score DataFrame} -> tidy table of mean ± sd across repeats."""
    rows = []
    for name, df in results.items():
        rows.append({"model": name, metric: df[metric].mean(), "sd_over_repeats": df[metric].std()})
    return pd.DataFrame(rows).set_index("model")


def grouped_permutation_importance(pipe, X, y, groups, task, n_splits=5, n_perm=20, seed=0):
    """
    Permutation importance measured on held-out papers. Each *raw* column is shuffled before
    the pipeline, so one-hot columns of the same variable are permuted together.
    Importance = increase in loss (MAE_log10 / Brier) when the column is shuffled.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    groups = np.asarray(groups)
    cv = GroupKFold(n_splits, shuffle=True, random_state=seed) if task == "reg" else \
        StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    imp = {c: [] for c in X.columns}

    def loss(est, Xte, yte):
        if task == "reg":
            return np.abs(est.predict(Xte) - yte).mean()
        return np.mean((est.predict_proba(Xte)[:, 1] - yte) ** 2)

    for tr, te in cv.split(X, y, groups):
        est = clone(pipe).fit(X.iloc[tr], y[tr])
        Xte, yte = X.iloc[te], y[te]
        base = loss(est, Xte, yte)
        for c in X.columns:
            d = []
            for _ in range(n_perm):
                Xp = Xte.copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                d.append(loss(est, Xp, yte) - base)
            imp[c].append(np.mean(d))
    return pd.DataFrame({"importance": {c: np.mean(v) for c, v in imp.items()},
                         "sd_over_folds": {c: np.std(v) for c, v in imp.items()}}
                        ).sort_values("importance", ascending=False)


# --------------------------------------------------------------------------- #
# 4. Plot style
# --------------------------------------------------------------------------- #
def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "legend.frameon": False, "font.size": 10,
    })
    warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# 5. Preprocessing + model zoo (fitted *inside* every CV fold -> no leakage)
# --------------------------------------------------------------------------- #
def make_preprocessor(num_cols, cat_cols, min_freq=4):
    """
    Numeric: median-impute (+ a 0/1 'was missing' flag per column) -> standardise.
    Categorical: 'not reported' for NaN -> one-hot; categories seen < min_freq times are pooled
    into one 'infrequent' bucket, so unseen reagents at test time never crash the model.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                       ("sc", StandardScaler())]), list(num_cols)))
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="not reported")),
                                       ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                            min_frequency=min_freq, sparse_output=False))]),
                      list(cat_cols)))
    return ColumnTransformer(parts, remainder="drop")


def regression_zoo(num_cols, cat_cols, seed=42, n_jobs=2):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    gkf = GroupKFold(3)                                    # inner folds also keep papers together
    mk = lambda est: Pipeline([("pre", make_preprocessor(num_cols, cat_cols)), ("m", est)])
    return {
        "Baseline: global median": DummyRegressor(strategy="median"),
        "Baseline: median per NP family": GroupMedianRegressor("np_family"),
        "Ridge": GridSearchCV(mk(Ridge()), {"m__alpha": [1, 10, 30, 100]},
                              cv=gkf, scoring="neg_mean_absolute_error", n_jobs=n_jobs),
        "Random forest": GridSearchCV(
            mk(RandomForestRegressor(n_estimators=200, random_state=seed)),
            {"m__min_samples_leaf": [1, 3, 6], "m__max_features": [0.3, 0.6, 1.0]},
            cv=gkf, scoring="neg_mean_absolute_error", n_jobs=n_jobs),
        "Gradient boosting": GridSearchCV(
            mk(HistGradientBoostingRegressor(max_iter=150, random_state=seed)),
            {"m__max_depth": [2, 3], "m__learning_rate": [0.05, 0.1], "m__min_samples_leaf": [3, 6]},
            cv=gkf, scoring="neg_mean_absolute_error", n_jobs=n_jobs),
    }


def classification_zoo(num_cols, cat_cols, seed=42, n_jobs=2):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    sgkf = StratifiedGroupKFold(3)
    mk = lambda est: Pipeline([("pre", make_preprocessor(num_cols, cat_cols)), ("m", est)])
    return {
        "Baseline: class prior": DummyClassifier(strategy="prior"),
        "Baseline: prior per NP family": GroupPriorClassifier("np_family"),
        "Logistic regression": GridSearchCV(
            mk(LogisticRegression(max_iter=2000, class_weight="balanced")), {"m__C": [0.03, 0.1, 0.3, 1]},
            cv=sgkf, scoring="neg_brier_score", n_jobs=n_jobs, error_score=np.nan),
        "Random forest": GridSearchCV(
            mk(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=seed)),
            {"m__min_samples_leaf": [1, 3, 6], "m__max_features": ["sqrt", 0.6]},
            cv=sgkf, scoring="neg_brier_score", n_jobs=n_jobs, error_score=np.nan),
    }


# --------------------------------------------------------------------------- #
# 6. How much of the variation is "which paper" vs "what the recipe changed"?
# --------------------------------------------------------------------------- #
def icc_between_papers(y, groups):
    """ICC(1): share of variance in y that is explained by *which paper a row comes from*."""
    d = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)})
    k = d.groupby("g").size()
    n, a = len(d), len(k)
    gm = d.groupby("g")["y"].transform("mean")
    grand = d["y"].mean()
    msb = ((gm - grand) ** 2).sum() / (a - 1)
    msw = ((d["y"] - gm) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def within_paper_effects(df, features, target, group, n_boot=1000, seed=0):
    """
    'What happens when one paper changes a knob?'  Subtract each paper's own mean from the feature and from the
    target, keep papers where the feature actually varies, and compute the Spearman correlation of the centred
    values. This removes lab, instrument and reporting differences between papers. The 95% interval resamples papers.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    out = []
    for f in features:
        d = df[[f, target, group]].dropna().copy()
        d["x"] = d[f] - d.groupby(group)[f].transform("mean")
        d["t"] = d[target] - d.groupby(group)[target].transform("mean")
        spread = d.groupby(group)["x"].transform(lambda s: s.abs().max())
        d = d[spread > 1e-9]
        parts = [(g.x.to_numpy(), g.t.to_numpy()) for _, g in d.groupby(group)]
        if len(parts) < 4:
            out.append({"feature": f, "within_paper_rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                        "n_papers": len(parts), "n_rows": len(d)})
            continue
        rho = spearmanr(d.x, d.t)[0]
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(parts), len(parts))
            x = np.concatenate([parts[i][0] for i in idx]); t = np.concatenate([parts[i][1] for i in idx])
            if np.ptp(x) > 0 and np.ptp(t) > 0:
                boots.append(spearmanr(x, t)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"feature": f, "within_paper_rho": rho, "ci_low": lo, "ci_high": hi,
                    "n_papers": len(parts), "n_rows": len(d)})
    return pd.DataFrame(out).set_index("feature")
