
import numpy as np
import sklearn



# CLASSIFICATION


def split_adata(adata, train_frac=0.85):
    """
        Split ``adata`` into train and test annotated datasets.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        train_frac: float
            Fraction of observations (cells) to be used in training dataset. Has to be a value between 0 and 1.

        Returns
        -------
        train_adata: :class:`~anndata.AnnData`
            Training annotated dataset.
        test_adata: :class:`~anndata.AnnData`
            Test annotated dataset.
    """
    train_size = int(adata.shape[0] * train_frac)
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_data = adata[train_idx, :]
    test_data = adata[test_idx, :]

    return train_data, test_data


def train_classifiers(X, labels, PCs, gene_ind):
    '''
    Trains a series of machine learning classifiers to associate individual cells with class labels.
    Does so in a low-dimensional PCA representation of the data (PCs) over pre-defined genes (gene_ind).
    '''  

    # Subset by gene indices; project X into PCA subspace
    X_ind = X[:,gene_ind]
    PCs_ind = PCs[gene_ind,:]
    X_PCA = np.matmul(X_ind,PCs_ind)
    
    # Specify classifiers and their settings 
    classifier_names = ['NearestNeighbors', 'SVM-Linear', 'SVM-RBF', 'DecisionTree', 'RandomForest', 
                        'NeuralNet', 'NaiveBayes', 'LDA']
    classifiers = [sklearn.neighbors.KNeighborsClassifier(20, weights='distance', metric='correlation'),
                   sklearn.svm.SVC(kernel='linear', gamma='scale', C=1, random_state=802),
                   sklearn.svm.SVC(kernel='rbf', gamma='scale', C=1, random_state=802),
                   sklearn.tree.DecisionTreeClassifier(random_state=802),
                   sklearn.ensemble.RandomForestClassifier(n_estimators=200, random_state=802),
                   sklearn.neural_network.MLPClassifier(random_state=802),
                   sklearn.naive_bayes.GaussianNB(),
                   sklearn.discriminant_analysis.LinearDiscriminantAnalysis()]
    
    # Split data into training and test subsets
    X_train, X_test, labels_train, labels_test = sklearn.model_selection.train_test_split(X_PCA, labels, test_size=0.5, random_state=802)
        
    # Build a dictionary of classifiers
    scores = []
    ClassifierDict={}
    for n,name in enumerate(classifier_names):
        clf_test = classifiers[n].fit(X_train, labels_train)
        score = clf_test.score(X_test, labels_test)
        scores.append(score)
        print(name,round(score,3))
        ClassifierDict[name]=classifiers[n].fit(X_PCA, labels)
    
    # Export classifier dictionary and subspace projection objects

    return {'Classes' : np.unique(labels),
            'Classifiers' : ClassifierDict,
            'Classifier_Scores' : dict(zip(classifier_names, scores)), 
            'PC_Loadings' : PCs,
            'Gene_Ind' : gene_ind}
   

def predict_classes(adata, Classifier):    
    '''
    '''
    X = adata.X
    X[np.isnan(X)]=0
    PCs = Classifier['PC_Loadings']
    gene_ind = Classifier['Gene_Ind']

    # First check to see if genes match between adata and Classifier 
    adata_genes = np.array(adata.var.index) 
    classifier_genes = np.array(gene_ind.index)
    if len(classifier_genes)==len(adata_genes):
        if (classifier_genes==adata_genes).all():
            # Subset by gene indices; project X into PCA subspace
            X_ind = X[:,gene_ind]
            PCs_ind = PCs[gene_ind,:]
            X_PCA = np.matmul(X_ind,PCs_ind)
    
    else:
        # Match highly variable classifier genes to adata genes, correcting for case
        adata_genes = np.array([x.upper() for x in adata_genes])
        classifier_genes = np.array([x.upper() for x in np.array(classifier_genes[gene_ind])])
        # Get overlap
        gene_overlap, dataset_ind, classifier_ind = np.intersect1d(adata_genes,classifier_genes,return_indices=True)
        # Subset by gene indices; project X into PCA subspace
        PCs_ind = PCs[gene_ind,:]
        PCs_ind = PCs_ind[classifier_ind,:]
        X_ind = X[:,dataset_ind]
        X_PCA = np.matmul(X_ind,PCs_ind)

    # Predict class labels and probabilities for each cell, store results in adata
    for n,name in enumerate(Classifier['Classifiers']):
        adata.obs['pr_'+name] = Classifier['Classifiers'][name].predict(X_PCA)
        if hasattr(Classifier['Classifiers'][name], "predict_proba"): 
            adata.obsm['proba_'+name] = Classifier['Classifiers'][name].predict_proba(X_PCA)

    return adata





