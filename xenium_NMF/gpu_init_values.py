import logging
import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpx_sparse
import rapids_singlecell as rsc
import scanpy as sc  # plotting / AnnData utilities
import matplotlib as mpl
import time
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------- helpers (unchanged math, GPU-aware replacements where needed) ----------

def G_a(mu, sd):
    return mu**2 / sd**2

def G_b(mu, sd):
    return mu / sd**2

def subset_cells(adata,
                 cells_per_category=5000,
                 stratify_category_key='sample'):
    adata.obs['_cell_index'] = np.arange(adata.n_obs)
    subset_ind = []
    for ct in adata.obs[stratify_category_key].unique():
        ind = adata.obs[stratify_category_key] == ct
        subset_ind_ = adata.obs['_cell_index'][ind]
        n_samples = np.min((len(subset_ind_), cells_per_category))
        subset_ind += list(np.random.choice(subset_ind_, size=n_samples, replace=False))
    n_cells_subset = len(subset_ind)
    print(f'Subsetted adata from {adata.shape[0]} to {n_cells_subset} cells')
    return adata[subset_ind, :].copy()

def rescale_distribution(dist: pd.DataFrame, n_factors: int):
    # `dist` is DataFrame or ndarray (cells x factors). We treat row-wise.
    dist = np.asarray(dist)
    q01 = np.quantile(dist, 0.01, axis=1).reshape((dist.shape[0], 1))
    mask = dist > q01
    dist = dist * mask + 0.01 - q01 * mask
    dist = (dist.T / dist.max(1)).T
    return dist

def max_min_sampling(data: pd.DataFrame, n_waypoints: int):
    # identical to your version (CPU; tiny)
    waypoint_set = []
    no_iterations = int((n_waypoints) / data.shape[1])
    N = data.shape[0]
    for ind in data.columns:
        vec = np.ravel(data[ind])
        iter_set = [np.random.randint(0, N)]
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = np.abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            min_dists = dists[:, 0:k].min(axis=1)
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)
            dists[:, k] = np.abs(vec - data[ind].values[new_wp])
        waypoint_set += iter_set
    waypoints = data.index[waypoint_set].unique()
    return waypoints

def _gpu_columnwise_mean_std(X_gpu_csr: cpx_sparse.csr_matrix):
    """
    Compute per-gene mean and std on GPU:
      mu = mean(X, axis=0)
      var = E[X^2] - (E[X])^2
    Returns cupy arrays (1D)
    """
    eps = 1e-8
    n_cells = X_gpu_csr.shape[0]
    # mean per col
    sums = cp.asarray(X_gpu_csr.sum(axis=0)).ravel()
    mu = sums / n_cells
    # E[X^2]
    X2 = X_gpu_csr.copy()
    X2.data = X2.data ** 2
    sums2 = cp.asarray(X2.sum(axis=0)).ravel()
    sq_mu = sums2 / n_cells
    var = sq_mu - mu ** 2
    std = cp.sqrt(cp.maximum(var, 0.0)) + eps
    return mu, std

def align_plot_stability(fac1, fac2, name1, name2, align=True, return_aligned=False, title=''):
    corr12 = np.corrcoef(fac1, fac2, False)
    ind_top = np.arange(0, fac1.shape[1])
    ind_right = np.arange(0, fac2.shape[1]) + fac1.shape[1]
    corr12 = corr12[ind_top, :][:, ind_right]
    corr12[np.isnan(corr12)] = -1
    if align:
        assignment = linear_sum_assignment(2 - corr12)[1]
        img = corr12[:, assignment]
    else:
        assignment = np.arange(corr12.shape[1])
        img = corr12
    plt.imshow(img)
    plt.title(f"{title}\n{name1} vs {name2}")
    plt.xlabel(name2)
    plt.ylabel(name1)
    plt.tight_layout()
    if return_aligned:
        return corr12, assignment

# ---------- GPU versions of your major steps ----------

def compute_pcs_knn_umap_gpu(
    adata_subset,
    fig_dir='',
    tech_category_key: Optional[str]=None,
    scale_max_value: int=10,
    n_comps: int=100,
    n_neighbors: int=15
):
    """
    GPU version of your 'compute_pcs_knn_umap':
      - log1p only (no normalize_total), matching your original
      - per-tech scaling (grouped z-score) implemented on GPU if tech_category_key is provided
      - RAPIDS PCA, drop PC1, RAPIDS neighbors+UMAP
    """
    # compute total_counts for plotting
    # Keep counts on GPU if possible
    rsc.get.anndata_to_GPU(adata_subset)

    # total_counts on GPU
    tc = cp.asarray(adata_subset.X.sum(axis=1)).ravel()
    adata_subset.obs['total_counts'] = cp.asnumpy(tc)

    # preserve raw counts in a layer
    adata_subset.layers['counts'] = adata_subset.X.copy()

    # log1p only (no normalize_total)
    rsc.pp.log1p(adata_subset)

    # scaling:
    if tech_category_key is None:
        # global scale on GPU
        rsc.pp.scale(adata_subset, max_value=scale_max_value)
    else:
        # grouped scale: compute group-wise mean/std on GPU and z-score
        cats = adata_subset.obs[tech_category_key].astype('category')
        adata_subset.obs[tech_category_key] = cats
        groups = cats.cat.categories.tolist()
        X_all = adata_subset.X  # csr on GPU
        # We'll write the scaled X back group-wise (in place)
        for g in groups:
            mask = (adata_subset.obs[tech_category_key].values == g)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            Xg = X_all[idx, :].copy()
            mu, std = _gpu_columnwise_mean_std(Xg)
            # (X - mu) / std, clipped to scale_max_value
            # sparse affine transform: X - mu -> subtract per-column means from nonzeros
            # For efficiency, do dense-ish route in chunks if needed; here we do direct:
            Xg = Xg.tocoo(copy=False)
            # subtract per-col mu
            Xg.data = Xg.data - mu[cp.asarray(Xg.col)]
            # divide by std
            Xg.data = Xg.data / std[cp.asarray(Xg.col)]
            # clip
            Xg.data = cp.clip(Xg.data, a_min=-scale_max_value, a_max=scale_max_value)
            Xg = Xg.tocsr()
            X_all[idx, :] = Xg
        adata_subset.X = X_all

    # PCA on GPU
    rsc.tl.pca(adata_subset, n_comps=n_comps)
    # QC plot: PC1 vs total_counts (on CPU plot)
    plt.hist2d(
        adata_subset.obsm['X_pca'][:, 0].astype(float),
        adata_subset.obs['total_counts'].values.astype(float),
        bins=200, norm=mpl.colors.LogNorm()
    )
    plt.xlabel('PC 1'); plt.ylabel('Total RNA count')
    plt.savefig(os.path.join(fig_dir, 'NMF_init_PC1_total_counts.pdf')); plt.close()

    # drop PC1
    adata_subset.obsm['X_pca'] = adata_subset.obsm['X_pca'][:, 1:]
    adata_subset.varm['PCs'] = adata_subset.varm['PCs'][:, 1:]

    # neighbors + UMAP on GPU
    rsc.pp.neighbors(adata_subset, n_neighbors=n_neighbors)  # uses X_pca by default
    rsc.tl.umap(adata_subset, min_dist=0.2, spread=0.8)      # same params as your scanpy call

    return adata_subset

def find_waypoint_gene_clusters_gpu(
    adata_neighbours,
    k='aver_norm',
    n_factors=300,
    margin_of_error=20,
    n_neighbors=15,
    labels_key=None,
    label_filter=None,
    verbose=True
):
    """
    GPU-based variant that uses PCs to represent genes (when labels_key is None).
    Builds a gene AnnData, runs GPU neighbors/UMAP for genes, then max–min waypoint selection.
    """
    if labels_key is not None:
        raise NotImplementedError("labels-based cluster averages not ported in this GPU version (can add if needed).")

    # Use PCs to represent genes, as in your original
    aver = pd.DataFrame(
        adata_neighbours.varm['PCs'],
        index=adata_neighbours.var_names,
        columns=[f'PC_{i+1}' for i in range(adata_neighbours.varm['PCs'].shape[1])],
    )
    gene_rates = {'aver': aver}
    gene_rates[k] = (gene_rates['aver'].T / gene_rates['aver'].abs().max(1)).T  # normalize by abs max per PC

    if verbose:
        print({kk: vv.shape for kk, vv in gene_rates.items()})

    # Build gene-level AnnData on CPU (lighter), but compute KNN/UMAP on GPU
    # We only need a tiny X so we don't move giant matrices unnecessarily.
    # Use tiny placeholder X and put representation into .obsm[k]
    gobs = adata_neighbours.var_names.copy()
    adata_neighbours_g = sc.AnnData(
        X=cp.asnumpy(cp.zeros((len(gobs), 1), dtype=cp.float32)),  # 1 dummy feature
        obs=pd.DataFrame(index=gobs)
    )
    # attach representation for neighbors
    adata_neighbours_g.obsm[k] = gene_rates[k].values

    # For coloring/size: log10 mean expression (from main adata) — compute on GPU, bring back
    rsc.get.anndata_to_GPU(adata_neighbours)
    X_cell_gene = adata_neighbours.X  # csr GPU
    mean_per_gene = cp.asnumpy(cp.asarray(X_cell_gene.mean(axis=0)).ravel())
    # else:
        # mean_per_gene = np.asarray(adata_neighbours.X.mean(axis=0)).ravel()
    adata_neighbours_g.obs['total_counts'] = np.log10(mean_per_gene + 1e-8)

    # neighbors on GPU using the representation
    # rsc.pp.neighbors can use 'use_rep' that points to obsm key
    rsc.pp.neighbors(adata_neighbours_g, n_neighbors=n_neighbors, use_rep=k, metric='correlation')
    rsc.tl.umap(adata_neighbours_g, min_dist=0.1, spread=2.5)

    # max–min sampling for waypoints (CPU)
    X_pd = pd.DataFrame(
        adata_neighbours_g.obsm[k],
        columns=[f"{k}_{i}" for i in range(adata_neighbours_g.obsm[k].shape[1])],
        index=adata_neighbours_g.obs_names,
    )
    init_n_factors = n_factors
    waypoints = max_min_sampling(data=X_pd, n_waypoints=n_factors)
    total_steps = 0
    while (abs(len(waypoints) - n_factors) > margin_of_error) and (total_steps <= 10):
        if verbose:
            print(len(waypoints), init_n_factors)
        waypoints = max_min_sampling(data=X_pd, n_waypoints=int(round(init_n_factors)))
        init_n_factors += 1.0 * (n_factors - len(waypoints))
        total_steps += 1
    n_factors = len(waypoints)

    # annotate
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs_names.isin(waypoints)
    adata_neighbours_g.obs["is_waypoint_size"] = np.array([10 if x else 1 for x in adata_neighbours_g.obs["is_waypoint"]])
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs["is_waypoint"].astype("category")

    return adata_neighbours_g, n_factors

def compute_w_initial_waypoint_gpu(
    adata_neighbours,
    adata_neighbours_g,
    n_factors,
    fig_dir='',
    k='aver_norm',
    scale=False, tech_category_key=None,
    use_x=True, layer=None,
    knn_smoothing=True,
    scale_max_value=10
):
    """
    GPU version of your W initialisation:
      W_init = X @ K_gene^T (averaged over waypoint genes' KNN neighborhood)
      optional KNN smoothing over cells (using cell connectivities)
    Everything stays on GPU where feasible.
    """
    # Ensure both on GPU
    if not rsc.get.is_anndata_on_GPU(adata_neighbours):
        rsc.get.anndata_to_GPU(adata_neighbours)
    if not rsc.get.is_anndata_on_GPU(adata_neighbours_g):
        rsc.get.anndata_to_GPU(adata_neighbours_g)

    # Choose X
    if use_x and layer is None:
        X = adata_neighbours[:, adata_neighbours_g.obs_names].X  # (cells x genes) CSR on GPU
    elif layer is not None:
        X = adata_neighbours[:, adata_neighbours_g.obs_names].layers[layer]
    else:
        X = adata_neighbours[:, adata_neighbours_g.obs_names].X

    # Waypoint mask (on CPU index, convert to GPU boolean mask)
    wp_mask_cpu = adata_neighbours_g.obs["is_waypoint"].values.astype(bool)
    wp_mask_gpu = cp.asarray(wp_mask_cpu)

    # gene connectivities (genes x genes) from gene graph
    K_gene = adata_neighbours_g.obsp["connectivities"]  # GPU CSR

    # select waypoint rows of K_gene
    K_wp = K_gene[wp_mask_cpu, :]  # (n_wp x n_genes)
    # average each gene by its waypoint neighborhood: X @ K_wp^T, then divide by row-sum
    denom = cp.asarray(K_wp.sum(axis=1)).ravel()  # (n_wp,)
    denom = cp.where(denom == 0, 1.0, denom)

    # compute W_init_raw = (X @ K_wp.T) / denom
    # X: (n_cells x n_genes), K_wp.T: (n_genes x n_wp) -> (n_cells x n_wp)
    W_init_gpu = (X @ K_wp.T).tocsr()
    # divide columns by denom
    W_init_gpu = W_init_gpu.tocoo(copy=False)
    W_init_gpu.data = W_init_gpu.data / denom[cp.asarray(W_init_gpu.col)]
    W_init_gpu = W_init_gpu.tocsr()

    # optional KNN smoothing over cells using cell connectivities
    if knn_smoothing:
        K_cell = adata_neighbours.obsp["connectivities"]  # (n_cells x n_cells) GPU CSR
        denom_cell = cp.asarray(K_cell.sum(axis=1)).ravel()
        denom_cell = cp.where(denom_cell == 0, 1.0, denom_cell)
        W_init_gpu = (K_cell @ W_init_gpu).tocsr()
        W_init_gpu = W_init_gpu.tocoo(copy=False)
        W_init_gpu.data = W_init_gpu.data / denom_cell[cp.asarray(W_init_gpu.row)]
        W_init_gpu = W_init_gpu.tocsr()

    # to dense (cells x n_wp) on GPU for rescaling + save histogram
    W_dense = W_init_gpu.toarray()  # cupy dense
    W_dense_cpu = cp.asnumpy(W_dense)  # small(ish) copy to CPU for rescale + plotting

    # rescale: bottom 1% -> 0, max -> 1 (row-wise)
    W_rescaled = rescale_distribution(W_dense_cpu, n_factors=W_dense_cpu.shape[0]).astype(np.float32)  # returns np array (cells x wp)

    # store in adata_neighbours.uns like original
    w_init_dict = {'cell_factors_w_cf': pd.DataFrame(
        W_rescaled, index=adata_neighbours.obs_names, columns=[f'factor_{i}' for i in range(W_rescaled.shape[1])]
    )}
    adata_neighbours.uns.setdefault('mod_init', {})
    adata_neighbours.uns['mod_init']['initial_values'] = {'w_init': w_init_dict}

    # histogram plot
    plt.hist(W_rescaled.ravel(), bins=500)
    plt.xlabel('Cell loading'); plt.ylabel('Frequency')
    plt.savefig(os.path.join(fig_dir, 'histogram_init_cell_loadings.pdf')); plt.close()

    return adata_neighbours

def find_stable_waypoint_gene_clusters_gpu(
    adata_neighbours,
    n_factors=300,
    fig_dir='',
    k='aver_norm',
    n_neighbors=20,
    labels_key='cell_type',
    n_repeats=5,
    cluster_max_cutoff=0.2,
    margin_of_error=20,
    bootstrap_p=0.9,
    verbose=True
):
    """
    GPU port of your stability bootstrapping (keeps CPU alignment/plots).
    """
    np.random.seed(1)
    adata_list = []
    for i in range(n_repeats):
        from sklearn.model_selection import train_test_split
        ind_, _ = train_test_split(
            np.arange(adata_neighbours.n_obs),
            test_size=1 - bootstrap_p,
            shuffle=True,
            stratify=adata_neighbours.obs[labels_key],
        )
        adata_sub = adata_neighbours[ind_, :].copy()
        adata_g, _ = find_waypoint_gene_clusters_gpu(
            adata_neighbours=adata_sub,
            k=k, n_factors=n_factors, margin_of_error=margin_of_error,
            n_neighbors=n_neighbors, labels_key=None, verbose=verbose
        )
        adata_list.append(adata_g.copy())

    # compute max correlation per cluster across bootstraps (CPU)
    cluster_max = np.zeros((adata_list[0].obs["is_waypoint"].values.astype(bool).sum()))
    for i in range(len(adata_list) - 1):
        c0 = np.array(adata_list[0].obsp["connectivities"][adata_list[0].obs["is_waypoint"].values.astype(bool), :].T.toarray())
        c1 = np.array(adata_list[i+1].obsp["connectivities"][adata_list[i+1].obs["is_waypoint"].values.astype(bool), :].T.toarray())
        corr01, assignment = align_plot_stability(
            fac1=c0, fac2=c1, name1='0', name2=f'{i+1}', title='Bootstrap step',
            align=True, return_aligned=True
        )
        if verbose:
            plt.savefig(os.path.join(fig_dir, 'align_plot_stability.pdf')); plt.close()
        else:
            plt.close()
        cluster_max = np.array([cluster_max, corr01.max(1)]).mean(0)

    plt.hist(cluster_max, bins=20)
    plt.savefig(os.path.join(fig_dir, 'cluster_max.pdf')); plt.close()

    waypoints = adata_list[0].obs_names[adata_list[0].obs["is_waypoint"].values.astype(bool)]
    waypoints = waypoints[cluster_max > cluster_max_cutoff]
    n_factors = len(waypoints)

    adata_neighbours_g = adata_list[0].copy()
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs_names.isin(waypoints)
    adata_neighbours_g.obs["is_waypoint_size"] = np.array([10 if x else 1 for x in adata_neighbours_g.obs["is_waypoint"]])
    adata_neighbours_g.obs["is_waypoint"] = adata_neighbours_g.obs["is_waypoint"].astype("category")

    with mpl.rc_context({'figure.figsize': [6, 6]}):
        sns.scatterplot(
            x=adata_neighbours_g.obsm["X_umap"][:,0],
            y=adata_neighbours_g.obsm["X_umap"][:,1],
            hue=adata_neighbours_g.obs['is_waypoint'],
            s=adata_neighbours_g.obs['is_waypoint_size']
        )
        plt.savefig(os.path.join(fig_dir, 'scatter_X_umap_waypoint.pdf')); plt.close()

    return adata_neighbours_g, n_factors

# ---------- main entrypoint (API-compatible with your original) ----------

def find_initial_values(
    adata,
    n_factors: int,
    stratify_category_key: str,
    tech_category_key: Optional[str],
    fig_dir='',
    cells_per_category: int = 100000
) -> Dict[str, np.ndarray]:
    """
    GPU-accelerated drop-in replacement.
    Returns:
        {'cell_factors_w_cf': (n_cells, n_factors) float32} – same as your original.
    """
    logging.info('find_initial_values[gpu] : .obs_names_make_unique()')
    adata_neighbours = adata.copy()
    adata_neighbours.uns['mod'] = {'gene_names': np.array(adata.var.index)}
    adata_neighbours.obs_names_make_unique()

    # Move X (cell-gene count matrix) to GPU
    rsc.get.anndata_to_GPU(adata_neighbours)

    # --- Step 1.0: subset
    logging.info(f'find_initial_values[gpu] : subset_cells(cells_per_category={cells_per_category}, stratify="{stratify_category_key}")')
    np.random.seed(1)
    adata_subset = subset_cells(
        rsc.get.to_anndata_CPU_view(adata_neighbours) if rsc.get.is_anndata_on_GPU(adata_neighbours) else adata_neighbours,
        cells_per_category=cells_per_category,
        stratify_category_key=stratify_category_key
    )
    # Move subset to GPU
    rsc.get.anndata_to_GPU(adata_subset)
    adata_neighbours = adata_subset.copy()

    # --- Step 1.1: PCA/Neighbors/UMAP on GPU (drop PC1)
    logging.info('find_initial_values[gpu] : compute_pcs_knn_umap_gpu()')
    adata_subset = compute_pcs_knn_umap_gpu(
        adata_subset,
        tech_category_key=tech_category_key,
        scale_max_value=10,
        n_comps=n_factors,
        n_neighbors=25,
        fig_dir=fig_dir
    )

    # --- Step 2.0: waypoint gene clusters on GPU
    logging.info('find_initial_values[gpu] : find_waypoint_gene_clusters_gpu()')
    adata_subset_g, n_factors = find_waypoint_gene_clusters_gpu(
        adata_subset,
        k='aver_norm',
        n_factors=n_factors,
        margin_of_error=20,
        n_neighbors=10,
        labels_key=None,
        label_filter=None,
        verbose=True
    )

    # enforce CSR (GPU)
    if not isinstance(adata_subset.X, cpx_sparse.csr_matrix):
        adata_subset.X = cpx_sparse.csr_matrix(adata_subset.X)
    if not isinstance(adata_subset_g.X, cpx_sparse.csr_matrix):
        adata_subset_g.X = cpx_sparse.csr_matrix(adata_subset_g.X)

    # --- Step 3.0: compute W init via GPU KNN smoothing
    logging.info('find_initial_values[gpu] : compute_w_initial_waypoint_gpu()')
    # ensure same cell order
    adata_subset = adata_subset[adata_neighbours.obs_names, :].copy()
    adata_subset = compute_w_initial_waypoint_gpu(
        adata_subset,
        adata_subset_g,
        n_factors,
        scale=True,
        tech_category_key=tech_category_key,
        use_x=True,
        knn_smoothing=True,
        fig_dir=fig_dir
    )
    adata_neighbours.uns['mod_init'] = adata_subset.uns['mod_init'].copy()

    # materialize into .obs (like your original)
    adata_subset.obs = adata_subset.obs.copy()
    cf = adata_subset.uns['mod_init']['initial_values']['w_init']['cell_factors_w_cf']
    adata_subset.obs[cf.columns] = cf
    adata_subset.obs = adata_subset.obs.copy()

    # Final dict, aligned to full (subset) cell order, float32
    init_vals = adata_neighbours.uns['mod_init']['initial_values']['w_init']
    init_vals = {k: v.loc[adata_neighbours.obs_names, :].values.astype('float32') for k, v in init_vals.items()}
    return init_vals


def init_gpu_memory(managed_memory=False, pool_allocator=True, device_id=0):
    rmm.reinitialize(
        managed_memory=managed_memory,
        pool_allocator=pool_allocator,
        devices=device_id,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)

def main():
    # Parse input arguments
    args = parse_args()

    # Load Anndata object
    args_adata = '/lustre/scratch124/cellgen/bayraktar/kr23/projects/STAGE/webatlas/output/NMF/subsampled_NMF/STAGE__concatenated__raw_counts__subsampled_0_05.h5ad'
    adata = sc.read_h5ad(args_adata)

    # Subset Anndata to genes if provided with one
    args_genes = '/nfs/team283/kr23/projects/STAGE/webatlas/data/NMF/STAGE_ASD_susceptbility_genes.csv'
    if args_genes is not None:
        args_genes_header = None
        genes = pd.read_csv(args_genes,
                            header=args_genes_header).iloc[:,0].tolist()
        logging.info(f'Subsetting to #{len(genes)} genes : {genes}')
        shp = adata.shape
        adata = adata[:, [gene in genes for gene in adata.var_names]].copy()
        logging.info(f'Reduced from shape {shp} to {adata.shape}')

    # Initialise directory to write quality control figures
    args_fig_dir = '/nfs/team283/kr23/github/Xenium_NMF/figs'
    os.makedirs(args_fig_dir, exist_ok=True)

    # Move Anndata to GPU
    rsc.get.anndata_to_GPU(adata)

    # Find initial values

    # 4) Call the GPU pipeline (drop-in replacement)
    t1 = time.time()
    init_vals = find_initial_values(
        adata=adata,
        n_factors=n_factors,
        stratify_category_key=stratify_category_key,
        tech_category_key=tech_category_key,
        fig_dir=fig_dir,
        cells_per_category=cells_per_category,
    )
    elapsed = time.time() - t1
    print(f"Initialisation done in {elapsed:.2f}s")

    # 5) Inspect / use results
    W = init_vals["cell_factors_w_cf"]  # shape: (n_cells_subset, n_factors)
    print(f"W shape: {W.shape} (cells x factors)")

    # If you want the factors attached to obs (like in your original), they’re already in
    # `adata.obs['factor_0'...'factor_{n_factors-1}']` inside the subset used.
    # You can also save out any plots made in `fig_dir`.

    return init_vals

if __name__ == "__main__":
    # ---- one-liner to run ----
    init_gpu_memory(managed_memory=False, pool_allocator=True, device_id=0)

    # Set your paths/params here:
    init_vals = main(
        h5ad_path="/lustre/scratch124/cellgen/bayraktar/kr23/projects/STAGE/webatlas/output/NMF/subsampled_NMF/STAGE__concatenated__raw_counts__subsampled_0_05.h5ad",
        n_factors=11,
        stratify_category_key="section",  # or "sample"
        tech_category_key="section",      # batch key for grouped scaling
        cells_per_category=100_000,
        fig_dir="/nfs/team283/kr23/github/Xenium_NMF/figs"
    )

    # Access the matrix directly:
    W = init_vals["cell_factors_w_cf"]
