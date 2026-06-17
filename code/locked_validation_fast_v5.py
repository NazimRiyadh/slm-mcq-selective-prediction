
import os, glob, math, shutil, zipfile, warnings, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

INPUT_ROOT = os.environ.get("SLM_FEATURE_INPUT", "/kaggle/input/datasets/nazimriyadh1/features/features")
OUTPUT_DIR = os.environ.get("SLM_LOCKED_OUTPUT", "/kaggle/working/slm_locked_validation_outputs")
NORMALIZED_DIR = os.environ.get("SLM_NORMALIZED_FEATURES", "/kaggle/working/features_normalized")

SEEDS = [0, 1, 2, 3, 4]
PCA_DIMS = [16, 32, 64]
LOGISTIC_C_GRID = [0.1, 1.0, 10.0]
BOOTSTRAP_B = int(os.environ.get("SLM_BOOTSTRAP_B", "300"))

COVERAGES = [0.80, 0.60, 0.40]
ABSTAIN_RATE = 0.20

KNOWN_DATASETS = [
    "arc_challenge", "commonsenseqa", "hellaswag", "mmlu_all", "mmlu",
    "arc_easy", "openbookqa", "boolq", "banking77"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_header(msg):
    print("\n" + "="*90)
    print(msg)
    print("="*90)

def normalize_files(src_root, dst_root):
    """Copy files to working dir and keep both original and normalized aliases."""
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        bn = p.name
        if not (bn.endswith(".csv") or bn.endswith(".npz") or bn.endswith(".json")):
            continue

        # Always copy original name.
        shutil.copy2(p, dst_root / bn)
        copied += 1

        # Also copy normalized aliases for older naming styles.
        aliases = set()
        aliases.add(bn.replace("_meta_features.csv", "__features.csv"))
        aliases.add(bn.replace("_hidden_embeddings.npz", "__hidden.npz"))
        aliases.add(bn.replace(".hidden.npz", "__hidden.npz"))
        aliases.add(bn.replace(".features.csv", "__features.csv"))
        aliases.add(bn.replace(".meta.json", "__meta.json"))

        # collapse more than two trailing underscores before type suffix
        for a in list(aliases):
            a2 = re.sub(r"_{3,}(features\.csv|hidden\.npz|meta\.json)$", r"__\1", a)
            aliases.add(a2)

        for a in aliases:
            if a != bn:
                try:
                    shutil.copy2(p, dst_root / a)
                except Exception:
                    pass

    print("Normalized/copy files:", copied)
    print("All feature-like CSV:", len(list(dst_root.glob("*features.csv"))))
    print("All hidden-like NPZ:", len(list(dst_root.glob("*hidden*.npz"))))
    return str(dst_root)

def strip_type_suffix(name):
    bn = os.path.basename(name)
    suffixes = [
        "__features.csv", "_features.csv", ".features.csv",
        "__hidden.npz", "_hidden.npz", ".hidden.npz",
        "_hidden_embeddings.npz", "__hidden_embeddings.npz",
        "__meta.json", "_meta.json", ".meta.json",
        "_meta_features.csv", "__meta_features.csv",
    ]
    for s in suffixes:
        if bn.endswith(s):
            base = bn[:-len(s)]
            return re.sub(r"[_.]+$", "", base)
    return None

def parse_model_dataset(name):
    """
    Robust parser for:
      qwen25_7b__arc_challenge__features.csv
      qwen25_7b__arc_challenge___features.csv
      gemma2_9b_it__arc_challenge.features.csv
      qwen25_7b_arc_challenge_meta_features.csv
      qwen25_7b__mmlu_all__hidden.npz
    """
    base = strip_type_suffix(name)
    if base is None:
        return None, None

    base = re.sub(r"[_.]+$", "", base)
    # turn 3+ underscores in separator area into double separator
    base = re.sub(r"___+", "__", base)

    # Preferred double-underscore format.
    if "__" in base:
        parts = [p.strip("_.") for p in base.split("__") if p.strip("_.")]
        if len(parts) >= 2:
            ds = parts[-1]
            model = "__".join(parts[:-1])
            if ds in KNOWN_DATASETS:
                return model, ds
            # sometimes dot style gives model__dataset with dataset recognized by suffix
            for known in sorted(KNOWN_DATASETS, key=len, reverse=True):
                if ds.endswith(known):
                    return model, known

    # Fallback known dataset suffix.
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if base.endswith("_" + ds):
            model = base[:-(len(ds)+1)].strip("_.")
            return model, ds

    return None, None

def discover(root):
    root = Path(root)

    hidden_index = {}
    for hp in list(root.glob("*hidden*.npz")):
        model, ds = parse_model_dataset(hp.name)
        if model and ds:
            hidden_index[(model, ds)] = str(hp)

    meta_index = {}
    for mp in list(root.glob("*meta*.json")):
        model, ds = parse_model_dataset(mp.name)
        if model and ds:
            meta_index[(model, ds)] = str(mp)

    rows = []
    seen = set()
    for csv in sorted(root.glob("*features.csv")):
        model, ds = parse_model_dataset(csv.name)
        if model is None:
            continue
        key = (model, ds)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "model": model,
            "dataset": ds,
            "dataset_display": "mmlu" if ds == "mmlu_all" else ds,
            "features_csv": str(csv),
            "hidden_npz": hidden_index.get(key, ""),
            "meta_json": meta_index.get(key, ""),
            "has_hidden": key in hidden_index,
        })

    if not rows:
        raise FileNotFoundError(f"No parsable feature CSV files found in {root}")

    df = pd.DataFrame(rows).sort_values(["model", "dataset"]).reset_index(drop=True)
    return df

def is_deepseek(model):
    return "deepseek" in model.lower()

def find_correct_col(df):
    for c in ["correct", "is_correct", "answer_correct", "pred_correct", "label_correct", "correctness"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "correct" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No correctness column. Columns: " + ", ".join(df.columns))

def get_y(df):
    col = find_correct_col(df)
    y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
    y = (y > 0).astype(int)
    return y, col

def get_cheap_cols(df, correct_col):
    include = ["top", "max", "prob", "conf", "margin", "entropy", "score", "logprob", "logit", "option", "choice"]
    exclude = ["id", "idx", "index", "question", "prompt", "answer", "gold", "label", "target", "correct", "text", "dataset", "model"]
    numeric = [c for c in df.columns if c != correct_col and pd.api.types.is_numeric_dtype(df[c])]

    cols = []
    for c in numeric:
        lc = c.lower()
        if any(t in lc for t in include) and not any(t in lc for t in exclude):
            arr = pd.to_numeric(df[c], errors="coerce").values
            if np.nanstd(arr) > 1e-12:
                cols.append(c)

    if len(cols) < 2:
        cols = []
        for c in numeric:
            lc = c.lower()
            if not any(t in lc for t in exclude):
                arr = pd.to_numeric(df[c], errors="coerce").values
                if np.nanstd(arr) > 1e-12:
                    cols.append(c)

    if not cols:
        raise ValueError("No cheap feature columns.")
    return cols

def clean(X):
    return np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def load_hidden(path, n):
    if not path or not os.path.exists(path):
        return None, ""
    z = np.load(path, allow_pickle=True)
    candidates = []
    for k in z.files:
        arr = z[k]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 2 and arr.shape[0] == n:
            candidates.append((k, arr))
        elif arr.ndim == 3:
            if arr.shape[0] == n:
                candidates.append((k + ":last_layer", arr[:, -1, :]))
            elif arr.shape[1] == n:
                candidates.append((k + ":last_layer", arr[-1, :, :]))
    if not candidates:
        print("[WARN] No hidden array matched n for", path)
        return None, ""
    k, arr = candidates[0]
    return clean(arr), k

def split_indices(y, seed):
    y = np.asarray(y)
    n = len(y)
    counts = np.unique(y, return_counts=True)[1]
    if len(counts) < 2 or counts.min() < 3:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        ntr = int(0.60*n)
        nval = int(0.20*n)
        return idx[:ntr], idx[ntr:ntr+nval], idx[ntr+nval:], "shuffle_fallback"
    try:
        s1 = StratifiedShuffleSplit(n_splits=1, train_size=0.60, random_state=seed)
        tr, tmp = next(s1.split(np.zeros(n), y))
        ytmp = y[tmp]
        s2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed+991)
        va_rel, te_rel = next(s2.split(np.zeros(len(tmp)), ytmp))
        return tr, tmp[va_rel], tmp[te_rel], "stratified"
    except Exception:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        ntr = int(0.60*n)
        nval = int(0.20*n)
        return idx[:ntr], idx[ntr:ntr+nval], idx[ntr+nval:], "shuffle_fallback"

def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return np.nan

def auprc_failure(y_correct, score_correct):
    fail = 1 - np.asarray(y_correct)
    if len(np.unique(fail)) < 2:
        return np.nan
    try:
        return float(average_precision_score(fail, 1 - np.asarray(score_correct)))
    except Exception:
        return np.nan

def risk_at(y, s, cov):
    n = len(y)
    k = max(1, int(np.ceil(cov*n)))
    keep = np.argsort(-s)[:k]
    return float(np.mean(1 - y[keep]))

def aurc(y, s):
    n = len(y)
    if n < 2:
        return np.nan
    order = np.argsort(-s)
    err = 1 - y[order]
    cum = np.cumsum(err)
    k = np.arange(1, n+1)
    cov = k/n
    risk = cum/k
    return float(np.trapezoid(np.r_[risk[0], risk], np.r_[0.0, cov]))

def eaurc(y, s):
    a = aurc(y, s)
    o = aurc(y, y.astype(float))
    if np.isnan(a) or np.isnan(o):
        return np.nan
    return float(a - o)

def failure_precision_recall(y, s, abstain_rate=0.20):
    n = len(y)
    k = max(1, int(np.ceil(abstain_rate*n)))
    abst = np.argsort(s)[:k]
    failures = (y == 0)
    if failures.sum() == 0:
        return np.nan, np.nan
    return float(failures[abst].mean()), float(failures[abst].sum()/failures.sum())

def all_metrics(y, s):
    out = {
        "auroc_correct": safe_auc(y, s),
        "auprc_failure": auprc_failure(y, s),
        "aurc": aurc(y, s),
        "eaurc": eaurc(y, s),
    }
    for c in COVERAGES:
        out[f"risk_at_{int(c*100)}"] = risk_at(y, s, c)
    p, r = failure_precision_recall(y, s, ABSTAIN_RATE)
    out["failure_precision_at_20_abstain"] = p
    out["failure_recall_at_20_abstain"] = r
    return out

def bootstrap_ci(y, s, fn, B=300, seed=13):
    n = len(y)
    if n < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        ids = rng.integers(0, n, n)
        v = fn(y[ids], s[ids])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def lr_scores(Xtr, ytr, Xva, Xte, C):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=1000))
    ])
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]

def candidates(Xcheap, Xhidden, tr, va, te):
    out = [("confidence_option", "cheap", 0, Xcheap[tr], Xcheap[va], Xcheap[te])]
    if Xhidden is not None:
        max_dim = min(Xhidden.shape[1], len(tr)-1)
        for d0 in PCA_DIMS:
            d = min(d0, max_dim)
            if d < 2:
                continue
            sc = StandardScaler()
            Htr_s = sc.fit_transform(Xhidden[tr])
            Hva_s = sc.transform(Xhidden[va])
            Hte_s = sc.transform(Xhidden[te])
            pca = PCA(n_components=d, random_state=0)
            Htr = pca.fit_transform(Htr_s)
            Hva = pca.transform(Hva_s)
            Hte = pca.transform(Hte_s)
            out.append(("hidden_pca", f"hidden_pca_{d}", d, Htr, Hva, Hte))
            out.append(("cheap_plus_hidden_pca", f"cheap_plus_hidden_pca_{d}", d,
                        np.hstack([Xcheap[tr], Htr]), np.hstack([Xcheap[va], Hva]), np.hstack([Xcheap[te], Hte])))
    return out

def evaluate_condition(row):
    model = row.model
    dataset = row.dataset_display
    print_header(f"Locked analysis: {model} / {dataset}")
    df = pd.read_csv(row.features_csv)
    y, ycol = get_y(df)
    if len(np.unique(y)) < 2:
        print("[SKIP] single class")
        return [], []
    cheap_cols = get_cheap_cols(df, ycol)
    Xcheap = clean(df[cheap_cols].apply(pd.to_numeric, errors="coerce").values)
    Xhidden, hkey = load_hidden(row.hidden_npz, len(df))
    print(f"n={len(y)}, acc={y.mean():.3f}, failures={(y==0).sum()}, cheap_cols={len(cheap_cols)}, hidden={None if Xhidden is None else Xhidden.shape}")

    details, selected = [], []
    for seed in SEEDS:
        tr, va, te, split_mode = split_indices(y, seed)
        byfam = defaultdict(list)
        for fam, cand, pca_dim, Xtr, Xva, Xte in candidates(Xcheap, Xhidden, tr, va, te):
            for C in LOGISTIC_C_GRID:
                try:
                    vs, ts = lr_scores(Xtr, y[tr], Xva, Xte, C)
                    vm = all_metrics(y[va], vs)
                    tm = all_metrics(y[te], ts)
                    rec = {
                        "model": model, "dataset": dataset, "seed": seed, "split_mode": split_mode,
                        "family": fam, "candidate": cand, "C": C, "pca_dim": pca_dim,
                        "n": len(y), "accuracy": float(y.mean()), "n_correct": int((y==1).sum()), "n_failures": int((y==0).sum()),
                        "n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_failures": int((y[te]==0).sum()),
                        "analysis_group": "deepseek_stress_test" if is_deepseek(model) else "primary_protocol_compatible",
                        "cheap_cols": "|".join(cheap_cols), "hidden_key": hkey
                    }
                    for k, v in vm.items():
                        rec["val_" + k] = v
                    for k, v in tm.items():
                        rec["test_" + k] = v
                    details.append(rec)
                    byfam[fam].append((rec, ts.copy(), y[te].copy()))
                except Exception:
                    continue
        for fam, items in byfam.items():
            valid = [(r, s, yt) for r, s, yt in items if np.isfinite(r.get("val_auroc_correct", np.nan))]
            if not valid:
                continue
            valid.sort(key=lambda x: x[0]["val_auroc_correct"], reverse=True)
            rec, ts, yt = valid[0]
            out = dict(rec)
            out["selected_by"] = "validation_auroc_locked"
            ci_fns = {
                "test_auroc_correct": lambda yy, ss: safe_auc(yy, ss),
                "test_auprc_failure": lambda yy, ss: auprc_failure(yy, ss),
                "test_risk_at_80": lambda yy, ss: risk_at(yy, ss, 0.80),
                "test_risk_at_60": lambda yy, ss: risk_at(yy, ss, 0.60),
                "test_risk_at_40": lambda yy, ss: risk_at(yy, ss, 0.40),
                "test_aurc": lambda yy, ss: aurc(yy, ss),
                "test_eaurc": lambda yy, ss: eaurc(yy, ss),
                "test_failure_precision_at_20_abstain": lambda yy, ss: failure_precision_recall(yy, ss)[0],
                "test_failure_recall_at_20_abstain": lambda yy, ss: failure_precision_recall(yy, ss)[1],
            }
            for name, fn in ci_fns.items():
                lo, hi = bootstrap_ci(yt, ts, fn, B=BOOTSTRAP_B, seed=seed+101)
                out[name + "_ci_low"] = lo
                out[name + "_ci_high"] = hi
            selected.append(out)
    return details, selected

def aggregate(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mets = [
        "test_auroc_correct", "test_auprc_failure", "test_risk_at_80", "test_risk_at_60", "test_risk_at_40",
        "test_aurc", "test_eaurc", "test_failure_precision_at_20_abstain", "test_failure_recall_at_20_abstain"
    ]
    rows = []
    for fam, g in df.groupby("family"):
        rec = {"family": fam, "rows": len(g), "conditions": g[["model", "dataset"]].drop_duplicates().shape[0]}
        for m in mets:
            rec[m + "_mean"] = float(g[m].mean()) if m in g else np.nan
            rec[m + "_std"] = float(g[m].std()) if m in g else np.nan
        rows.append(rec)
    summary = pd.DataFrame(rows)

    deltas = []
    for fam in sorted(df.family.unique()):
        if fam == "confidence_option":
            continue
        p = df[df.family.isin(["confidence_option", fam])].pivot_table(index=["model", "dataset", "seed"], columns="family", values=mets, aggfunc="first")
        for m in mets:
            try:
                d = (p[(m, fam)] - p[(m, "confidence_option")]).dropna()
            except Exception:
                continue
            if len(d) == 0:
                continue
            rng = np.random.default_rng(1)
            boots = [np.mean(rng.choice(d.values, len(d), replace=True)) for _ in range(3000)]
            try:
                wp = float(wilcoxon(d.values).pvalue) if not np.allclose(d.values, 0) else 1.0
            except Exception:
                wp = np.nan
            deltas.append({
                "comparison": f"{fam} - confidence_option", "metric": m, "n_pairs": int(len(d)),
                "mean_delta": float(d.mean()), "median_delta": float(d.median()),
                "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
                "wilcoxon_p": wp, "positive_cases": int((d > 0).sum()), "negative_cases": int((d < 0).sum())
            })
    delta_df = pd.DataFrame(deltas)

    cond_rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        for fam in sorted(df.family.unique()):
            if fam == "confidence_option":
                continue
            p = g[g.family.isin(["confidence_option", fam])].pivot_table(index="seed", columns="family", values="test_auroc_correct", aggfunc="first")
            if "confidence_option" in p.columns and fam in p.columns:
                d = (p[fam] - p["confidence_option"]).dropna()
                cond_rows.append({"model": model, "dataset": dataset, "comparison": f"{fam} - confidence_option", "mean_delta_auroc": float(d.mean()) if len(d) else np.nan, "n_seed_pairs": int(len(d))})
    return summary, delta_df, pd.DataFrame(cond_rows)

def paired_cost_utility(df, fam="cheap_plus_hidden_pca"):
    """Compute per-1000 operational table only on paired model/dataset/seed rows."""
    if df.empty:
        return pd.DataFrame()
    p = df[df.family.isin(["confidence_option", fam])].pivot_table(
        index=["model", "dataset", "seed"],
        columns="family",
        values="test_risk_at_80",
        aggfunc="first"
    ).dropna()
    if p.empty or "confidence_option" not in p.columns or fam not in p.columns:
        return pd.DataFrame()
    base = float(p["confidence_option"].mean())
    comp = float(p[fam].mean())
    rows = [
        {"family": "confidence_option", "paired_conditions_seeded": len(p), "mean_risk_at_80": base,
         "expected_errors_among_800_answered_per_1000_queries": base*800,
         "additional_errors_avoided_vs_conf_option": 0.0},
        {"family": fam, "paired_conditions_seeded": len(p), "mean_risk_at_80": comp,
         "expected_errors_among_800_answered_per_1000_queries": comp*800,
         "additional_errors_avoided_vs_conf_option": (base-comp)*800}
    ]
    return pd.DataFrame(rows)

def main():
    print("INPUT_ROOT:", INPUT_ROOT)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("BOOTSTRAP_B:", BOOTSTRAP_B)
    root = normalize_files(INPUT_ROOT, NORMALIZED_DIR)
    cond = discover(root)
    cond.to_csv(os.path.join(OUTPUT_DIR, "discovered_conditions.csv"), index=False)
    print("\nDiscovered conditions:")
    print(cond[["model", "dataset", "dataset_display", "has_hidden"]].to_string(index=False))

    all_details, all_selected = [], []
    for row in cond.itertuples(index=False):
        d, s = evaluate_condition(row)
        all_details.extend(d)
        all_selected.extend(s)
        pd.DataFrame(all_details).to_csv(os.path.join(OUTPUT_DIR, "locked_all_candidate_details_partial.csv"), index=False)
        pd.DataFrame(all_selected).to_csv(os.path.join(OUTPUT_DIR, "locked_validation_selected_partial.csv"), index=False)

    details = pd.DataFrame(all_details)
    selected = pd.DataFrame(all_selected)
    details.to_csv(os.path.join(OUTPUT_DIR, "locked_all_candidate_details.csv"), index=False)
    selected.to_csv(os.path.join(OUTPUT_DIR, "locked_validation_selected.csv"), index=False)

    if selected.empty:
        raise RuntimeError("No selected results produced.")

    primary = selected[selected.analysis_group == "primary_protocol_compatible"].copy()
    stress = selected[selected.analysis_group == "deepseek_stress_test"].copy()
    primary.to_csv(os.path.join(OUTPUT_DIR, "locked_primary_selected.csv"), index=False)
    stress.to_csv(os.path.join(OUTPUT_DIR, "locked_deepseek_stress_selected.csv"), index=False)

    for name, frame in [("all", selected), ("primary", primary), ("deepseek_stress", stress)]:
        summary, deltas, cond_delta = aggregate(frame)
        summary.to_csv(os.path.join(OUTPUT_DIR, f"summary_{name}_locked.csv"), index=False)
        deltas.to_csv(os.path.join(OUTPUT_DIR, f"paired_deltas_{name}_locked.csv"), index=False)
        cond_delta.to_csv(os.path.join(OUTPUT_DIR, f"per_condition_deltas_{name}_locked.csv"), index=False)

    cost = paired_cost_utility(primary, "cheap_plus_hidden_pca")
    cost.to_csv(os.path.join(OUTPUT_DIR, "cost_utility_primary_per_1000.csv"), index=False)

    fail_cols = ["model", "dataset", "seed", "n", "accuracy", "n_correct", "n_failures", "n_train", "n_val", "n_test", "test_failures", "split_mode", "analysis_group"]
    selected[fail_cols].drop_duplicates().to_csv(os.path.join(OUTPUT_DIR, "failure_counts_by_condition_seed.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "README_RESULTS.md"), "w") as f:
        f.write("# Locked validation fast v5 outputs\n\n")
        f.write("Primary analysis excludes DeepSeek stress-test. Cost utility uses paired rows only.\n\n")
        try:
            f.write("## Primary summary\n\n")
            f.write(pd.read_csv(os.path.join(OUTPUT_DIR, "summary_primary_locked.csv")).to_markdown(index=False))
            f.write("\n\n## Paired cost utility\n\n")
            f.write(cost.to_markdown(index=False))
        except Exception:
            pass

    zip_path = "/kaggle/working/slm_locked_validation_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in glob.glob(os.path.join(OUTPUT_DIR, "*")):
            z.write(p, arcname=os.path.basename(p))

    print_header("DONE")
    print("Output zip:", zip_path)
    print("Primary selected rows:", len(primary))
    print("DeepSeek stress selected rows:", len(stress))
    print("\nPrimary summary:")
    print(pd.read_csv(os.path.join(OUTPUT_DIR, "summary_primary_locked.csv")).to_string(index=False))
    print("\nPaired cost utility:")
    print(cost.to_string(index=False))

if __name__ == "__main__":
    main()
