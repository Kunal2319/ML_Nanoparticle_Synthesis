"""
Robustness check for the nanoparticle size/shape models.

Re-creates the preprocessing of the original notebooks (mean/mode imputation,
label encoding) and compares the original random 80/20 split against
(a) trivial baselines and (b) a split that keeps all rows from the same
source paper together (GroupKFold on the 'Reference' column).

Usage:
    python scripts/robustness_check.py "data/Nanoparticles_Synthesis_Dataset - Chemical_Synthesis.xlsx" chem
    python scripts/robustness_check.py "data/Nanoparticles_Synthesis_Dataset - Green_Synthesis.xlsx" green
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42


def load_chem(path):
    df = pd.read_excel(path)
    df = df.rename(columns={"Temperature(Â°C)": "Temperature(°C)"})
    for c in ["Precursor Concentration", "Reducing Agent Concentration"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    cols = ["NanoParticles", "Precursor", "Precursor Concentration", "Unit", "Reducing Agent",
            "Reducing Agent Concentration", "Unit2", "Stabilizing Agent",
            "Stabilizing Agent Concentration", "Unit3", "Temperature(°C)", "Agitation",
            "Time(min)", "Average Size(nm)", "Method to record sizes", "Shape"]
    cols = [c for c in cols if c in df.columns]
    groups = df["Reference"]
    d = df[cols].copy()
    d = d.dropna(subset=["Reducing Agent"])
    groups = groups.loc[d.index]
    # unit columns may be named Unit/Unit.1/Unit.2 or Unit/Unit2/Unit3 depending on the export
    unit_cols = [c for c in d.columns if c.lower().startswith("unit")]
    for c in ["Precursor Concentration", "Reducing Agent Concentration"]:
        d[c] = d[c].fillna(d[c].mean())
    for c in unit_cols[:2]:
        d[c] = d[c].fillna(d[c].mode().iloc[0])
    sc = "Stabilizing Agent Concentration"
    d[sc] = d[sc].fillna(d[sc].mean()).where(d["Stabilizing Agent"].notna(), 0)
    d["Stabilizing Agent"] = d["Stabilizing Agent"].fillna("No Stabilizing Agent")
    if len(unit_cols) > 2:
        u3 = unit_cols[2]
        d[u3] = d[u3].fillna(d[u3].mode().iloc[0]).where(d["Stabilizing Agent"] != "No Stabilizing Agent", "0")
    d["Temperature(°C)"] = d["Temperature(°C)"].fillna(25)
    d["Agitation"] = d["Agitation"].fillna("No Stirring")
    d["Time(min)"] = d["Time(min)"].fillna(d["Time(min)"].mean())
    d["Average Size(nm)"] = pd.to_numeric(d["Average Size(nm)"], errors="coerce")
    d["Average Size(nm)"] = d["Average Size(nm)"].fillna(d["Average Size(nm)"].mean())
    m = "Method to record sizes"
    d[m] = d[m].fillna(d[m].mode().iloc[0])
    d["Shape"] = d["Shape"].fillna(d["Shape"].mode().iloc[0])
    return d, groups


def load_green(path):
    df = pd.read_excel(path)
    df = df.rename(columns={"Temperature(Â°C)": "Temperature(°C)"})
    df["Reactant:Extract(v/v)"] = pd.to_numeric(
        df["Reactant:Extract(v/v)"].astype(str).str.strip(), errors="coerce")
    df["Average Size(nm)"] = pd.to_numeric(df["Average Size(nm)"], errors="coerce")
    cols = ["NanoParticles", "Reactant", "Conc of Reactant(mmol/L)", "Extract(Reducing Agent)",
            "Reactant:Extract(v/v)", "Temperature(°C)", "Time(min)", "Average Size(nm)", "Shape"]
    d = df[cols].copy()
    d = d.dropna(subset=["Reactant"]).dropna(subset=["Extract(Reducing Agent)"])
    groups = df["Reference"].loc[d.index]
    for c in ["Conc of Reactant(mmol/L)", "Reactant:Extract(v/v)", "Time(min)", "Average Size(nm)"]:
        d[c] = d[c].fillna(d[c].mean())
    d["Temperature(°C)"] = d["Temperature(°C)"].fillna(25)
    d["Shape"] = d["Shape"].fillna(d["Shape"].mode().iloc[0])
    return d, groups


def encode(d):
    X = d.drop(columns=["Average Size(nm)", "Shape"]).copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X, d["Average Size(nm)"].values, d["Shape"].astype(str).values


def cv_scores(X, y, groups, make_model, metric, splitter):
    scores = []
    for tr, te in splitter.split(X, y, groups):
        m = make_model().fit(X.iloc[tr], y[tr])
        scores.append(metric(y[te], m.predict(X.iloc[te])))
    return np.mean(scores), np.std(scores)


def main(path, kind):
    d, groups = load_chem(path) if kind == "chem" else load_green(path)
    X, y_size, y_shape = encode(d)
    print(f"{kind}: {len(d)} rows, {groups.nunique()} source papers")

    # 1) original protocol: single random 80/20 split, seed 42
    Xtr, Xte, ytr, yte = train_test_split(X, y_size, test_size=0.2, random_state=SEED)
    for name, mk in [("RF", lambda: RandomForestRegressor(random_state=SEED)),
                     ("GBR", lambda: GradientBoostingRegressor(random_state=SEED)),
                     ("Mean baseline", lambda: DummyRegressor())]:
        p = mk().fit(Xtr, ytr).predict(Xte)
        print(f"[size | random 80/20 split] {name:14s} MAE={mean_absolute_error(yte, p):7.2f}  R2={r2_score(yte, p):6.2f}")
    Xtr, Xte, ytr, yte = train_test_split(X, y_shape, test_size=0.2, random_state=SEED)
    for name, mk in [("RF", lambda: RandomForestClassifier(random_state=SEED)),
                     ("SVM", lambda: SVC(random_state=SEED)),
                     ("Majority", lambda: DummyClassifier(strategy="most_frequent"))]:
        p = mk().fit(Xtr, ytr).predict(Xte)
        print(f"[shape | random 80/20 split] {name:14s} acc={accuracy_score(yte, p):.3f}")

    # 2) 5-fold CV: random rows vs. whole papers held out
    gkf, kf = GroupKFold(n_splits=5), KFold(n_splits=5, shuffle=True, random_state=SEED)
    # rows with no Reference ID are treated as their own single-row group
    g = np.where(groups.isna(), "missing_" + pd.Series(range(len(groups))).astype(str), groups.astype(str))
    for label, splitter in [("random 5-fold", kf), ("group 5-fold (by paper)", gkf)]:
        for name, mk in [("RF", lambda: RandomForestRegressor(random_state=SEED)),
                         ("GBR", lambda: GradientBoostingRegressor(random_state=SEED)),
                         ("Mean baseline", lambda: DummyRegressor())]:
            m, s = cv_scores(X, y_size, g, mk, r2_score, splitter)
            print(f"[size | {label:24s}] {name:14s} mean R2={m:6.2f} (sd {s:.2f})")
        for name, mk in [("RF", lambda: RandomForestClassifier(random_state=SEED)),
                         ("SVM", lambda: SVC(random_state=SEED)),
                         ("Majority", lambda: DummyClassifier(strategy="most_frequent"))]:
            m, s = cv_scores(X, y_shape, g, mk, accuracy_score, splitter)
            print(f"[shape | {label:24s}] {name:14s} mean acc={m:.3f} (sd {s:.3f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
