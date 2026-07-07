
import numpy as np
import scipy




# SPARSE MATRICES


def sparse_corr(X):
  N = X.shape[0]
  C=((X.T*X -(sum(X).T*sum(X)/N))/(N-1)).todense()
  V=np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
  X_corr = np.divide(C,V+1e-119)
  return X_corr


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



