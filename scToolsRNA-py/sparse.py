
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



