# https://www.nxn.se/p/actionable-scrna-seq-clusters

#!/usr/bin/env python

# ----------
# Title:       simulate_data.py
# Details:     Koen Rademaker (kr23@sanger.ac.uk), 14 August 2025
# Function:    Generate a large synthetic AnnData object for NMF initialisation tests.
#              - Creates (n_cells x n_genes) count matrix with cluster structure
#              - Adds Control/Treated conditions
#              - (Optional) Introduces duplicate obs_names
#              - Saves to .h5ad
#              - (Optional) Runs xenium_NMF.utils.find_initial_values and saves .pkl
#
# Changes:
# * 0.1        14 August 2025
# - Base version
#
# Usage:
#   python 00_synthesize_anndata.py \
#       -out_h5ad /path/to/synthetic_1M.h5ad \
#       -out_pkl /path/to/init_vals.pkl \
#       -run_init 0 \
#       -n_cells 1000000 \
#       -n_genes 1000 \
#       -n_markers 120 \
#       -fraction_B 0.5 \
#       -fold_change 5.0 \
#       -dup_fraction 0.03 \
#       -seed 42 \
#       -chunk 20000
# ----------

import argparse
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse

# Optional: only needed if you set -run_init 1
try:
    import xenium_NMF
    from xenium_NMF.utils import find_initial_values
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize a large AnnData for NMF initialisation tests.")
    req = p.add_argument_group("required named arguments")
    req.add_argument("-out_h5ad", type=str, help="Output path for synthetic AnnData (.h5ad).")
    p.add_argument("-out_pkl", type=str, default=None, help="Output path for init_vals (.pkl) if -run_init 1.")
    p.add_argument("-run_init", type=bool, default=True, help="Run find_initial_values (0/1). Default: True.")
    p.add_argument("-n_cells", type=int, default=1000000, help="Number of cells. Default: 1,000,000.")
    p.add_argument("-n_genes", type=int, default=1000, help="Number of genes. Default: 1000.")
    p.add_argument("-n_factors", type=int, default=11, help="Number of factors. Default: 11.")
    p.add_argument("-n_markers", type=int, default=120, help="Number of marker (bimodal) genes. Default: 120.")
    p.add_argument("-fraction_B", type=float, default=0.5, help="Fraction of cluster B cells. Default: 0.5.")
    p.add_argument("-fold_change", type=float, default=5.0, help="FC for marker genes in cluster B. Default: 5.0.")
    p.add_argument("-dup_fraction", type=float, default=0.03, help="Fraction of duplicated obs_names. Default: 0.03.")
    p.add_argument("-seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument("-chunk", type=int, default=20000, help="Chunk size for Poisson sampling. Default: 20000.")
    try:
        args = p.parse_args()
    except:
        sys.exit(0)
    if args.out_h5ad is None:
        p.error("Please provide -out_h5ad")
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    n_cells = args.n_cells
    n_genes = args.n_genes
    n_markers = args.n_markers
    fraction_cluster_B = args.fraction_B
    fold_change = args.fold_change
    dup_fraction = args.dup_fraction
    chunk = args.chunk

    print(f"[info] Generating synthetic data: cells={n_cells:,}, genes={n_genes:,}, markers={n_markers}, "
          f"fraction_B={fraction_cluster_B}, FC={fold_change}, dup_fraction={dup_fraction}, seed={args.seed}")

    # 1) Latent clusters
    labels = rng.choice(["A", "B"], size=n_cells, p=[1 - fraction_cluster_B, fraction_cluster_B])

    # 2) Per-gene baseline (heavy-tailed) and 3) marker set
    gene_baseline = rng.gamma(shape=1.0, scale=1.0, size=n_genes)  # (G,)
    marker_idx = rng.choice(n_genes, size=n_markers, replace=False)

    # 4) Per-cell size factors (library depth)
    size_factors = rng.lognormal(mean=0.0, sigma=0.3, size=n_cells)  # (C,)

    # 5) Prepare condition labels
    condition = rng.choice(["Control", "Treated"], size=n_cells)

    # 6) Chunked Poisson sampling to control memory.
    # NOTE: This creates dense chunks, then converts to CSR and stacks.
    # With 1e6 x 1000, memory footprint is large even in sparse form if the matrix is not very sparse.
    # Adjust chunk if you hit memory pressure.
    is_B = (labels == "B")
    X_csr_chunks = []
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        # lambda for chunk: outer product of size_factors[start:end] and gene_baseline
        lam = np.multiply.outer(size_factors[start:end], gene_baseline)  # (chunk, G)
        # Upregulate marker genes in cluster B within the chunk
        if np.any(is_B[start:end]):
            lam[np.ix_(is_B[start:end], marker_idx)] *= fold_change
        # Sample counts
        X_block = rng.poisson(lam=lam).astype(np.int32, copy=False)
        # Convert to CSR and append
        X_csr_chunks.append(sparse.csr_matrix(X_block))
        # Free dense block ASAP
        del lam, X_block

        if (start // chunk) % 10 == 0:
            print(f"[progress] Generated {end:,} / {n_cells:,} cells")

    X = sparse.vstack(X_csr_chunks, format="csr")
    del X_csr_chunks

    # 7) obs / var tables
    obs = pd.DataFrame(
        {
            "cluster_truth": labels,
            "condition": condition,
            "size_factor": size_factors,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {
            "gene_symbol": [f"Gene{i}" for i in range(n_genes)],
            "is_marker": np.isin(np.arange(n_genes), marker_idx),
            "baseline_rate": gene_baseline,
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    # 8) Create AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(f"[info] AnnData created: X={adata.X.shape}, nnz={adata.X.nnz:,} "
          f"({adata.X.nnz / (n_cells*n_genes):.3%} non-zeros)")

    # Report counts per condition
    cond_counts = adata.obs["condition"].value_counts()
    n_control = int(cond_counts.get("Control", 0))
    n_treated = int(cond_counts.get("Treated", 0))
    print(f"[info] Cells per condition -> Control: {n_control:,} | Treated: {n_treated:,} | Total: {adata.n_obs:,}")

    # 9) Introduce duplicate obs_names (optional)
    if dup_fraction > 0:
        n_dups = int(dup_fraction * n_cells)
        victims = rng.choice(n_cells, size=n_dups, replace=False)
        donors = rng.choice(n_cells, size=n_dups, replace=True)
        new_names = adata.obs_names.to_numpy().copy()
        new_names[victims] = new_names[donors]
        adata.obs_names = new_names
        print(f"[warn] Introduced ~{dup_fraction*100:.1f}% duplicate obs_names. Unique? {adata.obs_names.is_unique}")

    # 10) Save to h5ad
    print(f"[info] Writing AnnData to: {args.out_h5ad}")
    adata.write_h5ad(args.out_h5ad, compression="gzip")
    print("[done] h5ad written.")

    # 11) Optional: run initialisation (very heavy at 1M cells; default off)
    if args.run_init:
        try:
            print("[info] Running find_initial_values (this can take a long time on 1M cells)...")
            init_vals = find_initial_values(
                adata,
                n_factors=args.n_factors,
                stratify_category_key="condition",
                tech_category_key="condition",
            )
            n_factors = init_vals[list(init_vals.keys())[0]].shape[1]
            print(f"[info] Initialised n_factors: {n_factors}")

            
            import pickle
            with open(args.out_pkl, "wb") as handle:
                pickle.dump(init_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[done] Saved init_vals to: {args.out_pkl}")
        except Exception as e:
            print(f"[error] find_initial_values failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()