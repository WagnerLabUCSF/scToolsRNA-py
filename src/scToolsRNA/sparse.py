
import numpy as np
import scipy




# SPARSE MATRICES


def sparse_corr(X):
  N = X.shape[0]
  C=((X.T*X -(sum(X).T*sum(X)/N))/(N-1)).todense()
  V=np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
  X_corr = np.divide(C,V+1e-119)
  return X_corr


def normalize_log1p(X, target_sum=1e6, log=True):
    """Library-size normalize each cell (row) to ``target_sum``, then ``log1p``.

    Shared low-level implementation of TPM-style (counts-per-``target_sum``)
    normalization followed by an optional natural-log ``log1p`` transform.
    Accepts a dense array or a scipy sparse matrix and returns the same kind;
    the input is not modified. Cells with zero total counts are left as
    all-zeros (no division).

    Used by :func:`scToolsRNA.pp_raw2norm`, :func:`scToolsRNA.adata2tpt`, and
    :func:`scToolsRNA.preprocess_query_counts` so the normalization math lives in
    exactly one place.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix
        Raw counts, cells x genes.
    target_sum : float, default ``1e6``
        Per-cell library size to scale to. ``1e6`` yields counts-per-million
        (TPM-scale) values; ``1e4`` yields CP10K.
    log : bool, default ``True``
        Apply ``log1p`` after normalization. Pass ``False`` to get the
        normalized-but-not-logged matrix (e.g. for a ``tpm_nolog`` layer).

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
        Normalized (and, if ``log``, log1p-transformed) matrix; sparse in,
        sparse out.
    """
    if scipy.sparse.issparse(X):
        X = X.tocsr(copy=True).astype(float)
        libsize = np.asarray(X.sum(axis=1)).ravel()
        scale = np.ones_like(libsize)
        nz = libsize > 0
        scale[nz] = target_sum / libsize[nz]
        X = (scipy.sparse.diags(scale) @ X).tocsr()
        if log:
            X.data = np.log1p(X.data)
        return X

    X = np.array(X, dtype=float, copy=True)
    libsize = X.sum(axis=1)
    scale = np.ones_like(libsize)
    nz = libsize > 0
    scale[nz] = target_sum / libsize[nz]
    X = X * scale[:, None]
    return np.log1p(X) if log else X


def convert_to_sparse(X):
  if not scipy.sparse.issparse(X):
    X=scipy.sparse.csr_matrix(X)
  return X


def convert_to_dense(X):
  if scipy.sparse.issparse(X):
    X=X.todense()
  return X


def filter_csr(matrix, keep=10):
    rows = matrix.shape[0]
    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(rows):
        row_data = matrix.data[matrix.indptr[i]:matrix.indptr[i+1]]
        row_indices = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
        smallest_indices = np.argpartition(row_data, keep)[:keep]
        
        new_data.extend(row_data[smallest_indices])
        new_indices.extend(row_indices[smallest_indices])
        new_indptr.append(len(new_data))
        
    return scipy.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape)



