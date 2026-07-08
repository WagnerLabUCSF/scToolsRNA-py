"""
kNN label transfer between single-cell datasets.

General-purpose tools for projecting cell annotations from a labeled reference
onto an unlabeled query, using distance-weighted k-nearest-neighbor voting in a
shared embedding (e.g. PCA or Harmony-corrected PCA). Includes:

- :func:`preprocess_query_counts` — TPM + log1p normalization of raw query counts.
- :func:`transfer_labels_knn`      — categorical label transfer with confidence scores.
- :func:`transfer_values_knn`      — continuous covariate transfer (e.g. pseudotime).

Adapted for general (organism-agnostic) use from the Wagner Lab ``zmap-tools``
package, whose kNN transfer was specialized for the zebrafish reference atlas.
The voting math (Gaussian/inverse distance weighting, inverse-frequency class
balancing, probability calibration) is preserved here; the atlas-specific
bookkeeping is not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .knn import knn_search
from .sparse import normalize_log1p

__all__ = [
    "preprocess_query_counts",
    "transfer_labels_knn",
    "transfer_values_knn",
]


def preprocess_query_counts(
    adata_query,
    *,
    counts_source="X",
    target_sum=1e6,
    inplace=True,
    integer_tol=1e-3,
    strict_counts=False,
):
    """
    Library-size normalize and log-transform raw counts for label transfer.

    Reads raw counts from the specified location, performs library-size
    normalization (``target_sum`` per cell, TPM-style by default) followed by
    ``log1p``, and writes the result into ``adata.X``. Intended to bring a query
    dataset onto the same scale as a log-normalized reference before projecting
    it and running :func:`transfer_labels_knn`.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset. Modified in place when ``inplace=True``.
    counts_source : str, default ``"X"``
        Where raw integer counts live. ``"X"`` uses ``adata.X``; any other value
        is treated as a layer name and reads ``adata.layers[counts_source]``.
    target_sum : float, default ``1e6``
        Library size each cell is scaled to before ``log1p``. ``1e6`` yields
        counts-per-million (TPM-scale) values.
    inplace : bool, default ``True``
        Modify ``adata_query`` in place (and return it) when ``True``; otherwise
        operate on and return a copy.
    integer_tol : float, default ``1e-3``
        Tolerance for the integer-like sanity check on the input counts.
    strict_counts : bool, default ``False``
        When ``True``, raise on NaN/inf, negative, or clearly non-integer input;
        when ``False``, emit a warning and continue.

    Returns
    -------
    anndata.AnnData
        The normalized AnnData (the same object when ``inplace=True``).
    """
    from scipy import sparse

    adata = adata_query if inplace else adata_query.copy()

    if counts_source == "X":
        X_raw = adata.X
        source_descr = ".X"
    else:
        if counts_source not in adata.layers:
            raise KeyError(
                f"counts_source='{counts_source}' not found in adata.layers. "
                "Use 'X' or a valid layer name."
            )
        X_raw = adata.layers[counts_source]
        source_descr = f"layers['{counts_source}']"

    if sparse.issparse(X_raw):
        data = X_raw.data
    else:
        X_raw = np.asarray(X_raw)
        data = X_raw.ravel()

    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Raw data in {source_descr} are not numeric.")

    finite_mask = np.isfinite(data)
    if not finite_mask.all():
        msg = "Raw counts contain NaN/inf values."
        if strict_counts:
            raise ValueError(msg)
        warnings.warn(msg)

    data_finite = data[finite_mask]
    if np.any(data_finite < 0):
        msg = "Raw counts contain negative values."
        if strict_counts:
            raise ValueError(msg)
        warnings.warn(msg)

    nonzero = data_finite[data_finite > 0]
    if nonzero.size > 0:
        sample = (
            nonzero
            if nonzero.size <= 1_000_000
            else np.random.default_rng(0).choice(nonzero, 1_000_000, replace=False)
        )
        frac = np.abs(sample - np.round(sample))
        if np.mean(frac > integer_tol) > 0.01:
            msg = (
                f"Raw data in {source_descr} do not appear integer-like "
                f"({np.mean(frac > integer_tol) * 100:.1f}% deviate > {integer_tol})."
            )
            if strict_counts:
                raise ValueError(msg)
            warnings.warn(msg)

    # ---- library-size normalization (TPM-ish) + log1p ----
    adata.X = normalize_log1p(X_raw, target_sum=target_sum)

    adata.uns["log1p"] = {"base": None}
    return adata


def _vote_weights(distances_row, scheme, sigma, inv_eps, inv_power, eps):
    """Per-neighbor weights from distances under the given weighting scheme."""
    if scheme is None:
        return np.ones(distances_row.shape[0], dtype=float)
    if scheme == "gaussian":
        s = sigma if (sigma is not None and sigma > 0) else (np.median(distances_row) + eps)
        return np.exp(-(distances_row * distances_row) / (2.0 * s * s))
    # "inverse"
    return 1.0 / np.power(distances_row + inv_eps, inv_power)


def transfer_labels_knn(
    adata_query,
    adata_ref,
    *,
    ref_label_col,
    ref_basis="X_pca_harmony",
    query_basis="X_pca_harmony",
    key_added=None,
    n_neighbors=25,
    metric="cosine",
    backend="auto",
    device="auto",
    nprobe=None,
    omit_labels=("unknown", "nan", "unassigned"),
    min_cells_per_label=15,
    vote_weighting="gaussian",
    vote_sigma=None,
    class_balance=None,
    balance_gamma=1.0,
    balance_eps=1e-9,
    inv_eps=1e-6,
    inv_power=1.0,
    p_thresh=0.8,
    apply_filter=True,
    inplace=True,
):
    """
    Transfer categorical labels from a reference to a query using kNN voting.

    For each query cell, finds its ``n_neighbors`` nearest reference cells in a
    shared embedding and tallies distance-weighted votes for each reference
    label. The winning label and its vote probability (confidence) are written
    into ``adata_query.obs``. Reference cells whose labels are missing, listed in
    ``omit_labels``, or belong to a class with fewer than ``min_cells_per_label``
    cells are dropped *before* the neighbor graph is built, so probabilities are
    computed only over eligible classes.

    Parameters
    ----------
    adata_query : anndata.AnnData
        Query dataset to annotate. Must contain ``query_basis`` in ``.obsm``.
    adata_ref : anndata.AnnData
        Labeled reference. Must contain ``ref_label_col`` in ``.obs`` and
        ``ref_basis`` in ``.obsm``. The two bases must share dimensionality
        (i.e. the query was projected into the reference embedding).
    ref_label_col : str
        Column in ``adata_ref.obs`` holding the labels to transfer.
    ref_basis, query_basis : str, default ``"X_pca_harmony"``
        ``.obsm`` keys giving the reference and query embeddings.
    key_added : str, optional
        Base name for output columns. Defaults to ``f"{ref_label_col}_predicted"``.
        Writes ``key_added`` (labels), ``key_added + "_prob"`` (top-vote
        probability), ``key_added + "_dist"`` (median neighbor distance), and
        the full probability matrix to ``adata_query.obsm[key_added + "_probs"]``.
    n_neighbors : int, default ``25``
        Neighbors per query cell.
    metric : {'cosine', 'euclidean'}, default ``"cosine"``
        Distance metric for the kNN search.
    backend, device, nprobe
        Passed through to :func:`scToolsRNA.knn.knn_search` (FAISS/sklearn).
    omit_labels : sequence of str or None, default ``("unknown", "nan", "unassigned")``
        Reference labels excluded from voting. Comparison is case-insensitive.
    min_cells_per_label : int, default ``15``
        Reference classes with fewer cells than this are excluded from voting.
    vote_weighting : {'gaussian', 'inverse', None}, default ``"gaussian"``
        Distance weighting for votes. ``"gaussian"`` (recommended) yields
        well-calibrated confidences; ``None`` gives uniform 1/k voting.
    vote_sigma : float, optional
        Gaussian kernel bandwidth. When ``None``, the per-cell median neighbor
        distance is used (adaptive).
    class_balance : {'global_inverse', None}, default ``None``
        If ``"global_inverse"``, upweight rare reference classes by inverse
        frequency (raised to ``balance_gamma``).
    balance_gamma : float, default ``1.0``
        Exponent on the inverse-frequency class weights.
    p_thresh : float or None, default ``0.8``
        Minimum top-vote probability to keep a prediction. Cells below are set to
        NaN in the main label column when ``apply_filter=True``.
    apply_filter : bool, default ``True``
        Apply the ``p_thresh`` gate to the main label column.
    inplace : bool, default ``True``
        Annotate ``adata_query`` in place (and return it) or a copy.

    Returns
    -------
    anndata.AnnData
        The annotated query (same object when ``inplace=True``).
    """
    if vote_weighting not in (None, "gaussian", "inverse"):
        raise ValueError("vote_weighting must be one of {None, 'gaussian', 'inverse'}.")
    if class_balance not in (None, "global_inverse"):
        raise ValueError("class_balance must be one of {None, 'global_inverse'}.")
    if ref_label_col not in adata_ref.obs:
        raise KeyError(f"ref_label_col '{ref_label_col}' not found in adata_ref.obs")
    if ref_basis not in adata_ref.obsm:
        raise KeyError(f"ref_basis '{ref_basis}' not found in adata_ref.obsm")
    if query_basis not in adata_query.obsm:
        raise KeyError(f"query_basis '{query_basis}' not found in adata_query.obsm")

    adata = adata_query if inplace else adata_query.copy()
    key = key_added or f"{ref_label_col}_predicted"

    # ---- reference filtering (omit BEFORE building the kNN index) ----
    ref_labels_full = adata_ref.obs[ref_label_col].astype(object)
    omit_lc = {str(x).lower() for x in omit_labels} if omit_labels else set()

    keep = ~ref_labels_full.isna()
    if omit_lc:
        keep &= ~ref_labels_full.map(lambda v: str(v).lower() in omit_lc)

    # drop rare classes
    if min_cells_per_label and min_cells_per_label > 0:
        counts = ref_labels_full[keep].astype(str).value_counts()
        rare = set(counts.index[counts < int(min_cells_per_label)])
        if rare:
            keep &= ~ref_labels_full.map(lambda v: str(v) in rare)

    n_keep = int(keep.sum())
    if n_keep < n_neighbors:
        raise ValueError(
            f"After excluding omit_labels/NaNs/rare classes, only {n_keep} reference "
            f"cells remain, fewer than n_neighbors={n_neighbors}. Reduce n_neighbors "
            "or relax filtering."
        )

    X_ref = np.asarray(adata_ref.obsm[ref_basis])[keep.values, :]
    ref_labels = ref_labels_full[keep].astype(str).to_numpy()
    X_query = np.asarray(adata_query.obsm[query_basis])

    # ---- neighbor graph ----
    neighbor_indices, distances, _meta = knn_search(
        X_ref,
        X_query,
        n_neighbors=n_neighbors,
        metric=metric,
        backend=backend,
        device=device,
        nprobe=nprobe,
    )

    # ---- class priors / balancing weights ----
    sorted_classes = np.sort(np.unique(ref_labels))
    C = len(sorted_classes)
    if C == 0:
        raise ValueError("No reference classes remain after filtering.")

    if class_balance == "global_inverse":
        cnt = pd.Series(ref_labels).value_counts().reindex(sorted_classes, fill_value=0)
        priors = cnt.to_numpy(dtype=float)
        priors = priors / (priors.sum() if priors.sum() > 0 else 1.0)
        w_class = np.power(priors + balance_eps, -balance_gamma)
        w_class = w_class / (w_class.mean() + balance_eps)
    else:
        w_class = np.ones(C, dtype=float)

    # ---- voting ----
    neighbor_classes = ref_labels[neighbor_indices]  # (n_query, k)
    probabilities = np.zeros((neighbor_indices.shape[0], C), dtype=float)
    has_votes = np.zeros(neighbor_indices.shape[0], dtype=bool)

    for i in range(neighbor_indices.shape[0]):
        valid = (neighbor_indices[i] >= 0) & np.isfinite(distances[i])
        if not np.any(valid):
            continue
        vals = neighbor_classes[i][valid]
        di = distances[i][valid]
        idxs = np.searchsorted(sorted_classes, vals)

        w_vote = _vote_weights(di, vote_weighting, vote_sigma, inv_eps, inv_power, balance_eps)
        if class_balance == "global_inverse":
            w_vote = w_vote * w_class[idxs]

        scores = np.bincount(idxs, weights=w_vote, minlength=C)
        s = scores.sum()
        if s > 0:
            probabilities[i, :] = scores / s
            has_votes[i] = True

    predicted = sorted_classes[np.argmax(probabilities, axis=1)].astype(object)
    predicted[~has_votes] = np.nan

    # ---- write outputs ----
    col_unfilt = f"{key}_unfilt"
    col_prob = f"{key}_prob"
    col_dist = f"{key}_dist"

    adata.obs[col_unfilt] = predicted
    adata.obs[key] = adata.obs[col_unfilt].copy()
    adata.obs[col_prob] = probabilities.max(axis=1)
    adata.obs[col_dist] = np.nanmedian(distances, axis=1)
    adata.obsm[f"{key}_probs"] = probabilities

    if apply_filter and p_thresh is not None:
        reject = ~(adata.obs[col_prob] >= p_thresh).fillna(False)
        adata.obs.loc[reject.values, key] = np.nan

    adata.uns.setdefault("scToolsRNA_labeltransfer", {})[key] = {
        "ref_label_col": ref_label_col,
        "ref_basis": ref_basis,
        "query_basis": query_basis,
        "n_neighbors": int(n_neighbors),
        "metric": metric,
        "vote_weighting": vote_weighting,
        "class_balance": class_balance,
        "p_thresh": (None if p_thresh is None else float(p_thresh)),
        "omitted_labels": list(omit_labels) if omit_labels else [],
        "min_cells_per_label": int(min_cells_per_label),
        "classes": sorted_classes.tolist(),
    }
    return adata


def transfer_values_knn(
    adata_query,
    adata_ref,
    *,
    ref_value_col,
    ref_basis="X_pca_harmony",
    query_basis="X_pca_harmony",
    key_added=None,
    n_neighbors=25,
    metric="cosine",
    backend="auto",
    device="auto",
    nprobe=None,
    stat="trimmed_mean",
    trim_alpha=0.25,
    distance_weighting="gaussian",
    sigma=None,
    inv_eps=1e-6,
    inv_power=1.0,
    eps=1e-9,
    inplace=True,
):
    """
    Transfer a continuous covariate from a reference to a query using kNN.

    Predicts a numeric value per query cell (e.g. developmental time or
    pseudotime) by aggregating the values of its nearest reference neighbors,
    optionally distance-weighted. Missing/non-numeric reference values are
    ignored per cell.

    Parameters
    ----------
    adata_query, adata_ref : anndata.AnnData
        Query and reference datasets, with embeddings in ``.obsm``.
    ref_value_col : str
        Numeric column in ``adata_ref.obs`` to transfer.
    ref_basis, query_basis : str, default ``"X_pca_harmony"``
        ``.obsm`` embedding keys.
    key_added : str, optional
        Output column name. Defaults to ``f"{ref_value_col}_predicted"``.
    n_neighbors : int, default ``25``
        Neighbors per query cell.
    metric : {'cosine', 'euclidean'}, default ``"cosine"``
        kNN distance metric.
    backend, device, nprobe
        Passed to :func:`scToolsRNA.knn.knn_search`.
    stat : {'trimmed_mean', 'mean', 'median', 'winsor_mean'}, default ``"trimmed_mean"``
        Aggregation function over neighbor values.
    trim_alpha : float, default ``0.25``
        Trim/winsor fraction in ``[0, 0.5)`` for the robust means.
    distance_weighting : {'gaussian', 'inverse', None}, default ``"gaussian"``
        Weighting of neighbor values by distance (ignored by ``median``).
    sigma : float, optional
        Gaussian bandwidth; per-cell median distance when ``None``.
    inplace : bool, default ``True``
        Write to ``adata_query`` in place (and return it) or a copy.

    Returns
    -------
    anndata.AnnData
        The annotated query (same object when ``inplace=True``).
    """
    if stat not in ("trimmed_mean", "mean", "median", "winsor_mean"):
        raise ValueError("stat must be one of {'trimmed_mean','mean','median','winsor_mean'}.")
    if not (0.0 <= float(trim_alpha) < 0.5):
        raise ValueError("trim_alpha must be in [0, 0.5).")
    if distance_weighting not in (None, "gaussian", "inverse"):
        raise ValueError("distance_weighting must be one of {None, 'gaussian', 'inverse'}.")
    if ref_value_col not in adata_ref.obs:
        raise KeyError(f"ref_value_col '{ref_value_col}' not found in adata_ref.obs")

    adata = adata_query if inplace else adata_query.copy()
    key = key_added or f"{ref_value_col}_predicted"

    ref_vals = pd.to_numeric(adata_ref.obs[ref_value_col], errors="coerce").to_numpy()
    X_ref = np.asarray(adata_ref.obsm[ref_basis])
    X_query = np.asarray(adata_query.obsm[query_basis])

    neighbor_indices, distances, _meta = knn_search(
        X_ref,
        X_query,
        n_neighbors=n_neighbors,
        metric=metric,
        backend=backend,
        device=device,
        nprobe=nprobe,
    )

    def _weighted_stat(v, w):
        if stat == "median" or w is None:
            if stat == "mean":
                return float(np.mean(v))
            if stat == "median":
                return float(np.median(v))
            # unweighted trimmed/winsor
            return _robust_mean(v, None, stat, trim_alpha, eps)
        if stat == "mean":
            return float((v * w).sum() / (w.sum() + eps))
        return _robust_mean(v, w, stat, trim_alpha, eps)

    out = np.full(neighbor_indices.shape[0], np.nan, dtype=float)
    for i in range(neighbor_indices.shape[0]):
        nbrs = neighbor_indices[i]
        v = ref_vals[nbrs]
        di = distances[i]
        ok = np.isfinite(v) & np.isfinite(di) & (nbrs >= 0)
        if not np.any(ok):
            continue
        v = v[ok]
        di = di[ok]
        if distance_weighting is None:
            w = None
        else:
            w = _vote_weights(di, distance_weighting, sigma, inv_eps, inv_power, eps)
            if not np.isfinite(w).any() or np.all(w == 0):
                w = None
        out[i] = _weighted_stat(v, w)

    adata.obs[key] = out
    return adata


def _robust_mean(x, w, stat, alpha, eps):
    """Trimmed or winsorized (optionally weighted) mean of a 1D array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if stat == "winsor_mean":
        lo, hi = np.quantile(x, [alpha, 1.0 - alpha])
        xw = np.clip(x, lo, hi)
        if w is None:
            return float(np.mean(xw))
        return float((xw * w).sum() / (np.sum(w) + eps))
    # trimmed_mean
    order = np.argsort(x)
    x_sorted = x[order]
    n = x_sorted.size
    k = int(np.floor(alpha * n))
    if n - 2 * k <= 0:
        return float(np.median(x_sorted))
    x_core = x_sorted[k : n - k]
    if w is None:
        return float(np.mean(x_core))
    w_core = np.asarray(w).ravel()[order][k : n - k]
    return float((x_core * w_core).sum() / (w_core.sum() + eps))
