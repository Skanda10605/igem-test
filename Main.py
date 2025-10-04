# main.py
# Consolidated and refactored script for iGEM project.
# This script combines data generation, model training, and prediction.

import os
import sys
import gc
import json
import random
import ast
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

# =========================
# Configuration
# =========================
# --- Input Files ---
PROMOTER_FILE = "hg19_promoter.txt"
EXPR_FILE = "BRCA_gene_exp_integrated.csv"
METH_FILE = "BRCA_meth_integrated_filtered.csv"
RANGE_FILE = "BRCA_data_meth_range.csv"
FILTERED_GENES_FILE = "integrated_gene_names_with_expression_new.csv"

# --- Output Files & Directories ---
MODEL_DIR = "Gene Wise Model Weights"
OUTPUT_DIR = "predictions_out"
MAPPED_OUT_CSV = "mapped_filteredgenes_data.csv"
LONG_OUT_CSV = "m_arrays_for_edit.csv"

# --- Model & Training Parameters ---
WINDOW_SIZE = 10_000_000
EPOCHS = 100      # MODIFIED: Reduced from 700 for faster testing
PATIENCE = 15     # MODIFIED: Reduced from 90 for faster testing
LR = 1e-3
VAL_FRAC = 0.2
SEED = 42

# --- Prediction & Analysis Parameters ---
PREDICTION_WINDOW_BP = 2000
SWEEP_WINDOW_START = 1000
SWEEP_WINDOW_END = 10000
SWEEP_WINDOW_STEP = 1000
SWEEP_LEVEL_START = 10.0
SWEEP_LEVEL_END = -10.0
SWEEP_LEVEL_STEP = 0.5

# =========================
# Setup
# =========================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device (GPU or CPU)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Using CPU")

# =========================
# Helper Functions
# =========================
def _standardize_chr(x: str) -> str:
    x = str(x).strip()
    return x if x.startswith("chr") else f"chr{x}"

def _guess_probe_col(df: pd.DataFrame):
    best, best_frac = None, 0.0
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        m = s.str.match(r"^cg\d+$", na=False)
        frac = m.mean()
        if frac > best_frac:
            best, best_frac = c, frac
    return best if best_frac >= 0.90 else None

def load_promoters_hg19(path: str) -> pd.DataFrame:
    prom = pd.read_csv(path, sep="\t", dtype={"chrID": str})
    prom = prom.rename(columns={"gene name": "gene", "chrID": "seqnames"})
    prom = prom[["gene", "seqnames", "start"]].copy()
    prom["gene"] = prom["gene"].astype(str).str.strip()
    prom["seqnames"] = prom["seqnames"].astype(str).apply(_standardize_chr)
    prom["start"] = pd.to_numeric(prom["start"], errors="coerce").astype("Int64")
    prom = prom.dropna(subset=["start"]).astype({"start": "int64"}).drop_duplicates()
    return prom

# =========================
# Model Architecture
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = x + res
        return self.act(x)

class AdaptiveRegressionCNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        out1 = min(64, input_size // 10)
        out2 = min(32, input_size // 20)
        out1 = max(out1, 1)
        out2 = max(out2, 1)

        self.conv1 = nn.Conv1d(1, out1, kernel_size=3, padding=1)
        self.res1  = ResidualBlock(out1)
        self.conv2 = nn.Conv1d(out1, out2, kernel_size=3, padding=1)
        self.res2  = ResidualBlock(out2)
        self.act   = nn.LeakyReLU(0.01)

        with torch.no_grad():
            d = torch.randn(1, 1, input_size)
            h = self.act(self.conv1(d))
            h = self.res1(h)
            h = self.act(self.conv2(h))
            h = self.res2(h)
            self._to_linear = h.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.res1(x)
        x = self.act(self.conv2(x))
        x = self.res2(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x).squeeze(-1)

# =========================
# Data Generation Function
# =========================
def generate_datasets_for_genes(target_genes: List[str]):
    print("--- Starting Data Generation ---")
    
    # Load all necessary files
    print("Loading promoter file...")
    prom = pd.read_csv(PROMOTER_FILE, sep="\t", dtype={"chrID": str})
    prom = prom.rename(columns={"gene name":"gene", "chrID":"seqnames"})[["gene","seqnames","start"]]
    prom["gene"] = prom["gene"].astype(str).str.upper().str.strip()
    prom["seqnames"] = prom["seqnames"].astype(str).map(_standardize_chr)
    prom["start"] = prom["start"].astype(int)

    print("Loading expression file...")
    expr = pd.read_csv(EXPR_FILE)
    expr["sample"] = expr["sample"].astype(str).str.upper().str.strip()
    expr = expr.set_index("sample").T

    print("Loading methylation matrix...")
    meth_raw = pd.read_csv(METH_FILE)
    probe_id_col = _guess_probe_col(meth_raw)
    if probe_id_col is None:
        raise ValueError("Could not auto-detect probe ID column in methylation matrix.")
    meth_raw[probe_id_col] = meth_raw[probe_id_col].astype(str).str.strip()
    meth_raw = meth_raw.drop_duplicates(subset=[probe_id_col], keep="first").reset_index(drop=True)
    meth = meth_raw.set_index(probe_id_col).T

    print("Loading CpG range file...")
    rng = pd.read_csv(RANGE_FILE)
    rng_probe_col = _guess_probe_col(rng)
    if rng_probe_col is None:
        rng_probe_col = "probe_id" if "probe_id" in rng.columns else rng.columns[0]
    rng = rng.rename(columns={rng_probe_col: "probe_id"})
    rng["probe_id"] = rng["probe_id"].astype(str).str.strip()
    rng["seqnames"] = rng["seqnames"].astype(str).map(_standardize_chr)
    rng["start"] = rng["start"].astype(int)

    # Sanity checks
    shared = set(meth.columns).intersection(set(rng["probe_id"]))
    if not shared:
        raise ValueError("No overlapping probe IDs between methylation matrix and range file.")

    cand = expr.index.intersection(meth.index)
    if len(cand) == 0:
        raise ValueError("No overlapping samples between expression and methylation.")
    
    sample_id = cand[0] # Use the first sample for consistency
    print(f"Using sample: {sample_id} for data generation")

    # Process genes
    targets_upper = [g.upper().strip() for g in target_genes]
    genes_with_prom = set(prom["gene"].unique())
    genes_with_expr = set(expr.columns.astype(str))
    selected_genes = [g for g in targets_upper if (g in genes_with_prom and g in genes_with_expr)]

    rng_by_chr = {k: v.sort_values("start").reset_index(drop=True) for k, v in rng.groupby("seqnames", sort=False)}
    prom_by_gene = prom.groupby("gene", sort=False)

    mapped_rows = []
    long_rows = []

    for gene in tqdm(selected_genes, desc="Generating data for genes"):
        gp = prom_by_gene.get_group(gene)

        chunks = []
        for _, r in gp.iterrows():
            chrn, tss = r["seqnames"], r["start"]
            dfc = rng_by_chr.get(chrn)
            if dfc is None:
                continue
            sel = dfc[(dfc["start"] >= tss - WINDOW_SIZE) & (dfc["start"] <= tss + WINDOW_SIZE)]
            if not sel.empty:
                chunks.append(sel)

        if not chunks:
            continue

        pw = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["probe_id"])
        pw = pw[pw["probe_id"].isin(meth.columns)]
        pw = pw.sort_values(["seqnames", "start"]).reset_index(drop=True)
        if pw.empty:
            continue

        probe_list = pw["probe_id"].tolist()
        coord_list = (pw["seqnames"].astype(str) + ":" + pw["start"].astype(str)).tolist()
        mvals = meth.loc[sample_id, probe_list].tolist()
        ge = float(expr.loc[sample_id, gene])

        mapped_rows.append({
            "gene": gene,
            "sample": sample_id,
            "gene_expression": ge,
            "cpg_m_values": mvals,
            "cpg_probe_ids": probe_list,
            "cpg_coords": coord_list
        })

        for pid, chrn, start, mv in zip(probe_list, pw["seqnames"], pw["start"], mvals):
            long_rows.append({
                "gene": gene,
                "sample": sample_id,
                "probe_id": pid,
                "chr": chrn,
                "start": int(start),
                "m_value": mv
            })

    pd.DataFrame(mapped_rows).to_csv(MAPPED_OUT_CSV, index=False)
    pd.DataFrame(long_rows).to_csv(LONG_OUT_CSV, index=False)

    print(f"\nWrote {MAPPED_OUT_CSV} with {len(mapped_rows)} genes.")
    print(f"Wrote {LONG_OUT_CSV} with {len(long_rows)} CpGs total.")
    print("--- Data Generation Complete ---")

# =========================
# Training Function
# =========================
def train_one_gene_and_save_weights(X, y, gene):
    idx_train, idx_val = train_test_split(
        np.arange(len(X)), test_size=VAL_FRAC, random_state=SEED, shuffle=True
    )
    Xtr, Xval = X[idx_train], X[idx_val]
    ytr, yval = y[idx_train], y[idx_val]

    Xtr = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1).to(device)
    ytr = torch.tensor(ytr, dtype=torch.float32).to(device) # FIX: Removed .unsqueeze(1)
    Xval = torch.tensor(Xval, dtype=torch.float32).unsqueeze(1).to(device)
    yval = torch.tensor(yval, dtype=torch.float32).to(device) # FIX: Removed .unsqueeze(1)

    input_size = X.shape[1]
    model = AdaptiveRegressionCNN(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    bad = 0

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(Xtr)
        loss = criterion(pred, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xval)
            vloss = criterion(vpred, yval).item()

        if vloss < best_val - 1e-9:
            best_val = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    safe_gene = gene.replace("/", "_").replace("\\", "_").replace(" ", "_")
    weight_path = os.path.join(MODEL_DIR, f"{safe_gene}.pt")
    torch.save(model.state_dict(), weight_path)

    meta = {
        "gene": gene,
        "input_size": int(input_size),
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "val_frac": VAL_FRAC,
        "device": str(device),
        "seed": SEED
    }
    with open(os.path.join(MODEL_DIR, f"{safe_gene}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    del model, optimizer, criterion, Xtr, Xval, ytr, yval
    torch.cuda.empty_cache()
    gc.collect()

def train_models_for_genes(genes_to_train: List[str]):
    print("--- Starting Model Training ---")
    
    print("--- Loading datasets for training ---")
    
    expr_df = pd.read_csv(EXPR_FILE)
    expr_df["sample"] = expr_df["sample"].astype(str).str.strip()
    gene_names_upper = expr_df["sample"].str.upper().str.strip().tolist()
    expr_values = expr_df.drop(columns=["sample"])
    expr_values.index = gene_names_upper
    expr = expr_values.T
    expr.index = expr.index.astype(str)

    meth_raw = pd.read_csv(METH_FILE)
    probe_col = "probe_id" if "probe_id" in meth_raw.columns else meth_raw.columns[0]
    probes = meth_raw[probe_col].astype(str).str.strip()
    meth_vals = meth_raw.drop(columns=[probe_col]).copy()
    meth_vals.columns = meth_vals.columns.astype(str)
    meth_vals.index = probes
    meth = meth_vals.T
    meth.index = meth.index.astype(str)
    meth.columns = meth.columns.astype(str)

    rng = pd.read_csv(RANGE_FILE)
    if "probe_id" not in rng.columns and "probeID" in rng.columns:
        rng = rng.rename(columns={"probeID": "probe_id"})
    rng = rng[["probe_id", "seqnames", "start"]].copy()
    rng["probe_id"] = rng["probe_id"].astype(str).str.strip()
    rng["seqnames"] = rng["seqnames"].astype(str).apply(_standardize_chr)
    rng["start"] = pd.to_numeric(rng["start"], errors="coerce").astype("Int64")
    rng = rng.dropna(subset=["start"]).astype({"start": "int64"})

    promoters = load_promoters_hg19(PROMOTER_FILE)
    promoters["gene"] = promoters["gene"].astype(str).str.upper().str.strip()
    
    genes_in_expr = set(expr.columns.tolist())
    user_genes_upper = set([g.upper().strip() for g in genes_to_train])
    eligible_genes = sorted(list(user_genes_upper & genes_in_expr & set(promoters["gene"].unique())))
    print(f"Eligible genes for training: {len(eligible_genes)}")
    
    if not eligible_genes:
        print("No eligible genes found. Exiting training.")
        return

    rng_by_chr = {k: v.sort_values("start").reset_index(drop=True) for k, v in rng.groupby("seqnames", sort=False)}

    for gene in tqdm(eligible_genes, desc="Training models"):
        gp = promoters[promoters["gene"] == gene]
        if gp.empty:
            continue

        probe_ids = []
        for chr_name, grp in gp.groupby("seqnames"):
            tss_list = grp["start"].tolist()
            df_chr = rng_by_chr.get(chr_name)
            if df_chr is None or df_chr.empty:
                continue
            s = df_chr["start"].values
            mask_all = np.zeros_like(s, dtype=bool)
            for tss in tss_list:
                w0, w1 = tss - WINDOW_SIZE, tss + WINDOW_SIZE
                mask_all |= ((s >= w0) & (s <= w1))
            if mask_all.any():
                probe_ids.append(df_chr.loc[mask_all, "probe_id"])

        if not probe_ids:
            continue

        probes_in_window = pd.Index(pd.concat(probe_ids).drop_duplicates().tolist())
        probes_in_window = probes_in_window.intersection(pd.Index(meth.columns))
        if probes_in_window.empty:
            continue

        use_samples = expr.index.intersection(meth.index)
        if use_samples.empty:
            continue

        gene_expr = expr.loc[use_samples, gene]
        meth_slice = meth.loc[use_samples, probes_in_window]
        X = meth_slice.values.astype(float)
        y = gene_expr.values.astype(float)

        if X.size == 0:
            continue
        na_ratio = np.isnan(X).sum() / X.size
        if na_ratio >= 0.20:
            continue
        X = KNNImputer(n_neighbors=5).fit_transform(X)

        try:
            train_one_gene_and_save_weights(X, y, gene)
        except Exception as e:
            print(f"[warn] Failed training {gene}: {e}")
            continue

        torch.cuda.empty_cache()
        gc.collect()

    print(f"--- Model Training Complete. Weights saved in: {MODEL_DIR} ---")


# =========================
# Prediction & Analysis Functions
# =========================

def resolve_input_probes(gene: str, long_df: pd.DataFrame) -> List[str]:
    gene_u = gene.upper()
    if os.path.exists(MAPPED_OUT_CSV):
        try:
            m = pd.read_csv(MAPPED_OUT_CSV)
            mg = m[m["gene"].astype(str).str.upper() == gene_u]
            if not mg.empty and "cpg_probe_ids" in mg.columns:
                raw = mg.iloc[0]["cpg_probe_ids"]
                out = ast.literal_eval(str(raw))
                out = [str(v) for v in out if str(v) != "nan"]
                if out:
                    print(f"[probe-order] source = {MAPPED_OUT_CSV} (n={len(out)})")
                    return out
        except Exception as e:
            print(f"[probe-order] mapped csv parse warning: {e}")

    sub = long_df[long_df["gene"].astype(str).str.upper() == gene_u].copy()
    if sub.empty:
        raise ValueError(f"Could not resolve input_probes for {gene_u}: no rows in long_df.")
    sub = sub.drop_duplicates("probe_id").sort_values(["chr", "start"])
    out = sub["probe_id"].astype(str).tolist()
    print(f"[probe-order] source = long_df genomic sort (n={len(out)})")
    return out

def pick_sample(df_long: pd.DataFrame, gene: str, sample: Optional[str]) -> str:
    df_g = df_long[df_long["gene"].str.upper() == gene.upper()]
    if df_g.empty:
        raise ValueError(f"No rows for gene {gene} in long data.")
    samples = sorted(df_g["sample"].astype(str).unique())
    if not samples:
        raise ValueError(f"No samples available for gene {gene}.")
    if sample is None:
        return samples[0]
    sample = str(sample)
    if sample not in samples:
        raise ValueError(f"Sample {sample} not found for {gene}. Available: {samples[:5]}...")
    return sample

def promoter_mask(input_probes: List[str], sub_long: pd.DataFrame, promoters: pd.DataFrame, gene: str, window_bp: int) -> np.ndarray:
    gene_u = gene.upper()
    pos = sub_long.drop_duplicates("probe_id").set_index("probe_id")[["chr", "start"]].to_dict("index")
    gp = promoters[promoters["gene"] == gene_u]
    if gp.empty:
        return np.zeros(len(input_probes), dtype=bool)

    gp_by_chr: Dict[str, List[int]] = {}
    for _, r in gp.iterrows():
        gp_by_chr.setdefault(str(r["seqnames"]), []).append(int(r["start"]))

    mask = np.zeros(len(input_probes), dtype=bool)
    for i, pid in enumerate(input_probes):
        info = pos.get(pid)
        if info is None:
            continue
        c, s = str(info["chr"]), int(info["start"])
        for tss in gp_by_chr.get(c, []):
            if (s >= tss - window_bp) and (s <= tss + window_bp):
                mask[i] = True
                break
    return mask

def closest_cpg_to_promoter(gene: str, promoters: pd.DataFrame, sub_long: pd.DataFrame) -> Dict:
    gene_u = gene.upper()
    gp = promoters[promoters["gene"] == gene_u]
    if gp.empty: return {}

    tsses = [(str(r["seqnames"]), int(r["start"])) for _, r in gp.iterrows()]
    best = {}
    best_abs = None

    for _, row in sub_long.drop_duplicates("probe_id").iterrows():
        c, s = str(row["chr"]), int(row["start"])
        same_chr = [t for t in tsses if t[0] == c]
        if not same_chr: continue
        dists = [abs(s - tss) for _, tss in same_chr]
        j = int(np.argmin(dists))
        d = int(dists[j])
        if (best_abs is None) or (d < best_abs):
            best_abs = d
            best = {
                "probe_id": str(row["probe_id"]), "chr": c, "start": s,
                "nearest_tss": same_chr[j][1], "distance_bp": d,
            }
    return best

def _predict_tensor(model: nn.Module, x_arr_1d: np.ndarray, device: str) -> float:
    x = torch.from_numpy(x_arr_1d).float().unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        y = model(x).cpu().numpy().reshape(-1)[0]
    return float(y)

def vector_from_long(df_long: pd.DataFrame, gene: str, sample: str, input_probes: List[str], fill_missing: float = 0.0) -> Tuple[np.ndarray, pd.DataFrame]:
    gene_u = gene.upper()
    sub = df_long[(df_long["gene"].str.upper() == gene_u) & (df_long["sample"] == sample)]
    if sub.empty:
        raise ValueError(f"No rows for {gene_u} / sample {sample} in long data.")
    sub = sub.drop_duplicates("probe_id").copy()
    m_by_probe = dict(zip(sub["probe_id"].astype(str), sub["m_value"]))
    x = np.array([m_by_probe.get(pid, fill_missing) for pid in input_probes], dtype=np.float32)
    return x, sub

def run_predictions_for_gene(gene: str):
    print(f"\n--- Running Predictions and Analysis for {gene} ---")
    
    df_long = pd.read_csv(LONG_OUT_CSV)
    df_long["gene"] = df_long["gene"].astype(str).str.upper().str.strip()
    df_long["sample"] = df_long["sample"].astype(str)
    df_long["chr"] = df_long["chr"].astype(str).map(_standardize_chr)
    df_long["start"] = pd.to_numeric(df_long["start"], errors="coerce").astype("Int64")
    df_long = df_long.dropna(subset=["start"]).astype({"start": "int64"})

    sample_id = pick_sample(df_long, gene, sample=None)
    print(f"Using sample: {sample_id} for predictions")

    input_probes = resolve_input_probes(gene, df_long)
    x_base, sub = vector_from_long(df_long, gene, sample_id, input_probes)
    
    promoters = load_promoters_hg19(PROMOTER_FILE)
    
    safe_gene = gene.replace("/", "_").replace("\\", "_").replace(" ", "_")
    meta_path = os.path.join(MODEL_DIR, f"{safe_gene}.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    input_size = meta['input_size']

    model = AdaptiveRegressionCNN(input_size)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{safe_gene}.pt"), map_location=device))
    model.to(device).eval()

    if len(x_base) != input_size:
        if len(x_base) > input_size:
            x_base = x_base[:input_size]
        else:
            x_base = np.pad(x_base, (0, input_size - len(x_base)), 'constant')
    
    # Task 1: Basic Predictions
    pmask = promoter_mask(input_probes, sub, promoters, gene, window_bp=PREDICTION_WINDOW_BP)
    if len(pmask) != input_size:
        if len(pmask) > input_size: pmask = pmask[:input_size]
        else: pmask = np.pad(pmask, (0, input_size - len(pmask)), 'constant')

    x_prom_hyper = x_base.copy(); x_prom_hyper[pmask] = 10.0
    
    preds = {
        "baseline": _predict_tensor(model, x_base, device),
        "all_plus10": _predict_tensor(model, np.full_like(x_base, 10.0), device),
        "all_minus10": _predict_tensor(model, np.full_like(x_base, -10.0), device),
        "promoter_hyper_+10": _predict_tensor(model, x_prom_hyper, device),
    }
    df_task1 = pd.DataFrame([{"manipulation": k, "y_pred": v} for k, v in preds.items()])
    task1_csv = os.path.join(OUTPUT_DIR, f"predictions_{gene}_{sample_id}.csv")
    df_task1.to_csv(task1_csv, index=False)
    print(f"Saved Task 1 results to {task1_csv}")

    # Task 2: Sweep Window Hypomethylation
    rows_task2 = []
    for W in range(SWEEP_WINDOW_START, SWEEP_WINDOW_END + 1, SWEEP_WINDOW_STEP):
        pmask_sweep = promoter_mask(input_probes, sub, promoters, gene, window_bp=W)
        if len(pmask_sweep) != input_size:
            if len(pmask_sweep) > input_size: pmask_sweep = pmask_sweep[:input_size]
            else: pmask_sweep = np.pad(pmask_sweep, (0, input_size - len(pmask_sweep)), 'constant')

        x_edit = x_base.copy()
        x_edit[pmask_sweep] = -10.0
        y = _predict_tensor(model, x_edit, device)
        rows_task2.append({"window_bp": W, "y_pred": y})
    df_task2 = pd.DataFrame(rows_task2)
    
    plt.figure()
    plt.plot(df_task2["window_bp"], df_task2["y_pred"], marker="o")
    plt.xlabel("Promoter window size (±bp)")
    plt.ylabel("Predicted expression")
    plt.title(f"{gene}: Expression vs Hypo-methylated Window")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"plot_sweep_window_{gene}_{sample_id}.png"))
    plt.close()
    
    task2_csv = os.path.join(OUTPUT_DIR, f"sweep_window_hypo_{gene}_{sample_id}.csv")
    df_task2.to_csv(task2_csv, index=False)
    print(f"Saved Task 2 results and plot for {gene}")

    # Task 3: Sweep Promoter Level
    rows_task3 = []
    level = SWEEP_LEVEL_START
    while level >= SWEEP_LEVEL_END:
        x_edit = x_base.copy()
        x_edit[pmask] = level
        y = _predict_tensor(model, x_edit, device)
        rows_task3.append({"promoter_level": level, "y_pred": y})
        level -= SWEEP_LEVEL_STEP
    df_task3 = pd.DataFrame(rows_task3)

    plt.figure()
    plt.plot(df_task3["promoter_level"], df_task3["y_pred"], marker="o")
    plt.xlabel("Promoter CpG M-value")
    plt.ylabel("Predicted expression")
    plt.title(f"{gene}: Expression vs Promoter Methylation (±{PREDICTION_WINDOW_BP} bp)")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"plot_sweep_level_{gene}_{sample_id}.png"))
    plt.close()

    task3_csv = os.path.join(OUTPUT_DIR, f"sweep_promoter_level_{gene}_{sample_id}.csv")
    df_task3.to_csv(task3_csv, index=False)
    print(f"Saved Task 3 results and plot for {gene}")
    print(f"--- Predictions for {gene} Complete ---")

# =========================
# Main Execution Block
# =========================
if __name__ == "__main__":
    try:
        # Get user input for genes
        genes_input = input("Enter the gene(s) you want to train and analyze (comma-separated): ")
        user_genes = [gene.strip() for gene in genes_input.split(',')]

        if not user_genes:
            print("No genes provided. Exiting.")
            sys.exit(1)

        # Step 1: Generate necessary datasets
        generate_datasets_for_genes(user_genes)
        
        # Step 2: Train models for the specified genes
        train_models_for_genes(user_genes)
        
        # Step 3: Run predictions and generate plots for each trained gene
        for gene in user_genes:
            safe_gene = gene.replace("/", "_").replace("\\", "_").replace(" ", "_")
            if os.path.exists(os.path.join(MODEL_DIR, f"{safe_gene.upper()}.pt")):
                run_predictions_for_gene(gene.upper())
            else:
                print(f"Skipping predictions for {gene} as model training failed or was not eligible.")

        print("\n--- All tasks complete! ---")
        print(f"Check the '{MODEL_DIR}' directory for model weights.")
        print(f"Check the '{OUTPUT_DIR}' directory for prediction CSVs and plots.")

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found: {e.filename}")
        print("Please make sure all input data files are in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
