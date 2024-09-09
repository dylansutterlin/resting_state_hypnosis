import networkx as nx
import numpy as np
import sys
import os
import glob as glob
import pickle
import pandas as pd
import bct.algorithms as bct
from sklearn.model_selection import permutation_test_score
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.linear_model import RidgeCV
#from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.abspath('../scripts'))
from scripts import plot_func

def compute_graphs_metrics(connectomes, subjects, labels, out_type='dict', verbose = False): # random_graphs =True, n_permut = 50):
    """
    Compute graph metrics for each subject's graph
    and return a dict of networkX graphs, or a list of graphs if out_type = 'list'

    Parameters
    ----------
    connectomes : list of connectomes (adjacency matrices) for each subject
    subjects : list of strings, subjects names. If permut, input e.g. names = [f'perm{i} for i in range(n_permut)]
    labels : list of strings from the atlas ROIs
    out_type : str, 'dict' or 'list', default = 'dict'
        Put 'list' for easier manipulation of perm graphs metrics

    results : dict or list and dict of metrics containing list of vectors for each 'subject'

    """
    metric_dict = dict(
        subjects=subjects,
        nodes=labels
        )
    # Initialize single-subject graphs from Adjacency matrix (As)
    As = [connectomes[i] for i in range(len(connectomes))]
    rawGs = {nx.from_numpy_array(cm, create_using=nx.Graph) for cm in connectomes}
    rawGs = {
        nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels))) for G in rawGs
            } # assign labels to nodes for each dict in list
    # add keys (subjects names) to make it a dict instead od a set
    Gs = dict() # dict of graph/subjects
    for sub, G in zip(subjects, rawGs):
        Gs[sub] = G
    ls_Gs = [] # list of graphs

    # Adding a strengh, distance, centralities, clustering to each graph
        # Most code comes from Centeno et al., 2022. suited for abs(weighted graphs)
    for subject, graph in Gs.items():
        # strenght = degree but for weighted graphs, remove weight attribute for binary graphs
        strengths = {node: val for (node, val) in nx.degree(graph, weight="weight")}  # nx.degree return tuple; store as dict{node: val}
        nx.set_node_attributes(graph, strengths, "strength")
        norm_strengths = {
            node: val * 1 / (len(graph.nodes) - 1) for (node, val) in nx.degree(graph, weight="weight")
        } # normalization for easier interpretation
        G_distance_dict = {
            (e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data="weight")
        }  # convert weight to distance to compute betweenness centrality
        #nx.set_edge_attributes(graph, G_distance_dict, "d")
        nx.set_node_attributes(graph, norm_strengths, "strengthnorm")
        nx.set_edge_attributes(graph, G_distance_dict, "distance")
        #nx.set_node_attributes(graph, nx.degree(graph, weight="weight"), 'degree')
        nx.set_node_attributes(graph, nx.eigenvector_centrality(graph, weight = 'weight'), 'eigenCent')
        nx.set_node_attributes(graph, nx.betweenness_centrality(graph, weight="distance"), "betCentrality")
        nx.set_node_attributes(graph, nx.closeness_centrality(graph, distance="distance"), "closeCent")
        nx.set_node_attributes(graph, nx.degree_centrality(G), "degCentrality")
        nx.set_node_attributes(graph, nx.clustering(graph, weight="weight"), "clustering")
        nx.set_node_attributes(graph, nx.community.louvain_communities(graph), "community")
        localEff = bct.efficiency_wei(nx.to_numpy_array(graph, weight="weight")) # BCTpython, returns numpy array
        nx.set_node_attributes(graph, localEff, "localEfficiency")

        Gs[subject] = graph #update graph to list of graphs
        ls_Gs.append(graph) 
    
        metric_list = ['weight', 'strength', 'strengthnorm', 'distance', 'eigenCent', 'betCentrality', 'closecent', 'degCentrality', 'clustering', 'community', 'localEfficiency']
    
    if out_type == 'list' : # Use case for permutation, where keys (subnames) are not needed
        return ls_Gs, metric_list, metric_dict
    
    if verbose != False:
        print(r'[compute_graphs_metrics()] Done computing N = {} graphs and metrics for {} condition'.format(len(subjects), verbose))

    return Gs, metric_list, metric_dict

def node_attributes2df(Gs_dict, node_metrics):
    '''
    Return a list of dataframes for each subject, each dataframe contains the node attributes for each node
    
    Parameters
    ----------
    Gs_dict : dict of subjects containing list of graphs
    node_metrics : list of node attributes to be included in each dataframe
    
    return : list of dataframes, list of subjects/keys names
    '''

    ls_df = []
    keys_ls = []
    for sub, G in Gs_dict.items(): # sub is key, G is value (graph), otherwise a tuple
        nan_array = np.full((len(G.nodes), len(node_metrics)), np.nan)
        df = pd.DataFrame(nan_array, index=G.nodes, columns=node_metrics)

        for node in G.nodes:
            attributes = G.nodes[node] #dict for node i{'metric1' : int., 'm2':int.}

            for metric in node_metrics: # for node i, fix columns with metric
                df.loc[node, metric] = attributes[metric]

        ls_df.append(df)
        keys_ls.append(sub)

    return ls_df, keys_ls

def edge_attributes2df(Gs_dict, edge_metric = 'weight', subjects = False):
    ''' 
    Takes dict of networkX graphs, extracts its specified edge values.
    Returns a sub x pairs of nodes dataframe, with values being, e.g. weights
    
    Parameters
    ----------
    Gs_dict : dict
        Keys are subjects, or e.g. permutations, and items() are networkX graphs
    edge_metric : str.
        Name to pass to G.edges().data('str'). Default = 'weight'
        https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.edges.html
    subjects list
        list of sub to put as row names in output dfs. else, dict keys are used.
    
    return : pd.DataFrameS
    '''
    if subjects is False:
        subjects = Gs_dict.keys()
    #assert len(subjects) == Gs_dict.keys(), 'Subjects names and Gs_dict items do not match'

    tuple_ls = [G.edges().data(edge_metric) for G in Gs_dict.values()] # (node 1, node 2, value)
    edges_dfs = [] # to append sub i x pairs of single row dfs
    for i, sub in enumerate(subjects):
        l_nodes = []
        l_edges = []
        for u, v, edge in tuple_ls[i]:
            l_nodes.append(u+'-'+v) # pair of nodes
            l_edges.append(edge) # edge value
        edges_dfs.append(pd.DataFrame([np.array(l_edges)], columns = l_nodes, index = [sub]))    
    
    return pd.concat(edges_dfs, axis=0)


def rand_conmat(ls_connectomes, subjects, n_permut = 1000, algo= 'hqs'):
    """
    Randomize connectomes for each subject and returns a dict of every subjects containing list of Perm connectomes

    Parameters
    ----------
    ls_connectomes : list of connectomes (adjacency matrices) for each subject
    subjects : list of strings, subjects names
    n_permut : int, number of permutation for each subject

    * Code vaild only for symetric matrices. Else, compute mean (e) of upper and lower triangle
    return : dict of subjects containing list of Perm connectomes
    """
    perm_matrices = dict()
    for i, sub in enumerate(subjects):
        perms = []
        mat = ls_connectomes[i] 
        seeds = np.random.randint(0, 1000000, n_permut)

        for j in range(n_permut):

            N = mat.shape[0]
            e = np.mean(mat[np.tril_indices(N, k=-1)]) # mean of off-diag
            cov = np.sum((mat[np.triu_indices(N, k=1)] - e) ** 2) / (N * (N - 1))
            var = np.mean(np.diag(mat))
            rand_conmat = hqs_algorithm(e, cov, var, N, seeds[j], return_correlation=True)
            perms.append(rand_conmat)
        #print(N, e, cov, var, rand_conmat.min(), rand_conmat.max())
        perm_matrices[sub] = perms

    return perm_matrices
#
def hqs_algorithm(e, cov, var, N, seed, return_correlation=False):
    """
    HQS algorithm to compute the covariance matrix and correlation matrix of a graph
    See Zaleski et al., 2012 in Neuroimage for description and validation of the algo.
    Parameters
    ----------
    e : float, 
        Mean of off-diagonal elements of the matrix
    cov : float, 
        Variance of off-diagonal elements (covariance) of the matrix
    var : float, 
        Mean of diagonal elements of the matrix (variance)
    N : int, 
        Dimension of the matrix
    seed : float, 
        Seed for the random number generator
    return_correlation : bool, default = False
        If True, return the correlation matrix instead of the covariance matrix
    """

    np.random.seed(seed)

    # Step 2: Calculate m = m ← max(2, ⌊(ē2 − e2)/v⌋)
    m = max(2, int(np.floor((var**2 - e**2) / cov)))
    # Step 3: Calculate μ
    mu = np.sqrt(e / m)
    # Step 4: Calculate σ^2
    sigm = -(mu**2) + np.sqrt(mu**4 + cov / m)
    # Step 5-6: Generate random samples from Gaussian distribution
    X = np.random.normal(mu, np.sqrt(sigm), size=(N, m))
    # Step 7: Compute covariance matrix
    C = np.dot(X, X.T)
    # Step 8: Transform covariance matrix to correlation matrix using aCa where a :
    a = np.diag(1 / np.sqrt(np.diag(C)))
    correlation_matrix = np.dot(np.dot(a, C), a)

    if return_correlation:
        return correlation_matrix
    else:
        return C

   # Old code can remove if run success /11 march 24

    #resulting_covariance_matrix = hqs_algorithm(e, v, edash, N)
    #print("Covariance Matrix:")
    #print(resulting_covariance_matrix)

    #resulting_correlation_matrix = hqs_algorithm(e, v, edash, N, return_correlation=True)
    #print("\nCorrelation Matrix:")
    #print(resulting_correlation_matrix)#


    # ---Feature matrices based on subjects' graphs---#
    #subject_names = list(Gs.keys())
    #node_names = list(Gs[subject_names[0]].nodes())#

    # Reorganize data structure from lists of vectors to pd (N x nodes array) for each dict key
    #for metric in list(metric_dict.keys()):  # Excluding nodes and communities
    #    tmp_data = metric_dict[metric]
    #    if metric != 'communities' and metric != 'nodes':
    #        metric_dict[metric] = pd.DataFrame(
    #            np.array(tmp_data), columns=labels, index=subjects
    #        )
    ##breakpoint() # check data structure for each nodes ( clustering ?)
    ##maybe save as a df would be easier to compute 2samp t tests
    ##breakpoint()#store the yeo7 id of ROI tocompute network metrics/subjects

    #return metric_dict  # dict w/ keys = metrics and values = list of list vectors (one list per subject)

def rand_graphs(dict_graphs, subjects, n_permut):
    '''
    Randomize graphs for each subject and returns a dict of every subjects containing list of NetworkX graphs
    '''

    perm_graphs = dict()
    for i, sub in enumerate(subjects):
        perms = []
        seeds = np.random.randint(0, 1000000, n_permut)

    return perm_graphs


def node_metric_diff(post_df, pre_df, subjects):

    change_dfs = []
    for i, sub in enumerate(subjects):
        assert list(post_df[i].columns) == list(pre_df[i].columns)

        temp_df = post_df[i].copy(deep=True)
        temp_df = post_df[i] - pre_df[i]
        change_dfs.append(temp_df)

    return change_dfs

def bootstrap_pvals_df(df_metric_ls, dict_dfs_permut, subjects, mult_comp = 'fdr_bh'):
    '''
    Compute cell-wise p_values of each cell (node x metric/i,j) compared to
    i,jth disrtibution of permuted dataframes (node x metric x permuts)
    '''
    pval_dfs_ls = []
    pvalfdr_dfs_ls = []
    for i, sub in enumerate(subjects):
        df_metric = df_metric_ls[i]
        rand_dist_3d = np.stack(dict_dfs_permut[sub], axis=2) # stack list of 2d rand dfs on 3d dim
        assert df_metric.shape == rand_dist_3d.shape[0:2]
        #mean_val = np.mean(rand_dist, axis=2)
        #td_val = np.std(rand_dist, axis=2)

        p_df = np.zeros_like(df_metric, dtype=float)
        for i in range(df_metric.shape[0]):
            # Compute p-values for each cell compare to dist along 3d dim of permut
            for j in range(df_metric.shape[1]):
                cell_value = df_metric.iloc[i, j]
                p_df[i, j] = np.mean(np.abs(rand_dist_3d[i, j, :]) >= np.abs(cell_value)) # Two-tailed test

        p_values_df = pd.DataFrame(p_df, index=df_metric.index, columns=df_metric.columns)
        pval_dfs_ls.append(p_values_df)

        if mult_comp == 'fdr_bh':
            for column in p_values_df.columns:
                p_values = p_values_df[column]
                _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
                p_values_df[column] = corrected_p_values
        pvalfdr_dfs_ls.append(p_values_df)

    if mult_comp == 'fdr_bn':
        print('FDR correction applied to change metric p-values')

        return pval_dfs_ls, pvalfdr_dfs_ls
    
    return pval_dfs_ls


def connectome2feature_matrices(ls_connectomes, subjects):
    """
    Extracts half of the connectome matrix (lower triangular) and computes flattens it to stack it in array
    Connectomes : list of connectomes (adjacency matrices) for each subject
    return : array of shape (n_subjects, n_features)
    """
    # Lower triangular mask
    tril_mask = np.tril(np.ones(ls_connectomes[0].shape[-2:]), k=-1).astype(bool)
    X_con = np.stack(
        [ls_connectomes[i][..., tril_mask] for i in range(0, len(ls_connectomes))],
        axis=0,
    )  # store the original con matrix based on the flatten subjects' features (shape : N x features)

    return pd.DataFrame(X_con, index=subjects)


def compute_permutation(
    X,
    y,
    pipeline,
    gr=None,
    cv=None,
    n_components=0.80,
    n_permutations=5000,
    scoring="r2",
    rdm_seed=40
):
    """
    Compute the permutation test for a specified metric (r2 by default)
    Apply the PCA after the splitting procedure

    Parameters
    ----------
    X: predictive variable
    y: predicted variable
    n_components: number of components to keep for the PCA
    n_permutations: number of permuted iteration
    scoring: scoring strategy
    random_seed: controls the randomness

    Returns
    ----------
    score (float): true score
    perm_scores (ndarray): scores for each permuted samples
    pvalue (float): probability that the true score can be obtained by chance

    See also scikit-learn permutation_test_score documentation
    """
    if gr == None and cv == None:
        cv = KFold(n_splits=5, random_state=random_seed, shuffle=True)
    # ShuffleSplit(n_splits = 5, test_size = 0.3, random_state = random_seed)
    elif gr != None and cv == None:
        cv = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=random_seed)

    score, perm_scores, pvalue = permutation_test_score(
        estimator=pipeline,
        X=X,
        y=y,
        scoring=scoring,
        cv=cv,
        n_permutations=n_permutations,
        random_state=rdm_seed,
    )

    return score, perm_scores, pvalue

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd

def regression_cv(features_matrix, Y, target_columns, pred_metric_name = 'X_pred', rdm_seed=40, n_permut = 5000, test_size = 0.2, pca='80%', doPlot = False):

    """
    Compute the regression with cross-validation for each graph metric
    and return a dict of results
    features_matrix : Feature matrix N subjects x M features
    Y : dataframe, index = subject names and columns = target columns
    target_columns : list of strings from the target columns
    metrics_names : list of strings from the graph metrics but written in a way that will figure in the plots outputs.
    """

    mean_metrics = []
    #out_dict = dict()
    result_Yi = dict()

    prePipeline = Pipeline([("scaler", StandardScaler())])
    feat_mat = prePipeline.fit_transform(features_matrix)
    pipeline = Pipeline([("pca", PCA(n_components=pca)), ("reg", Lasso(alpha=1.0))])
    # cv = KFold(n_splits=5, random_state=randrdm_seedom_seed, shuffle=True)
    cv = ShuffleSplit(n_splits=5, test_size=test_size, random_state=rdm_seed)

    for target_column in target_columns:
        print(f"[CV-regression] : __{target_column} ~ -{pred_metric_name}__")
        y_preds = []
        y_tests = []
        y = Y[target_column].values

        pearson_r_scores = []
        pval_pearsonr_scores = []
        r2_scores = []
        mse_scores = []
        rmse_scores = []
        n_components = []
        all_coefficients = []
        all_corr_coefficients = []
        pca_ls = []
        pca_coeffs = []

        for train_index, test_index in cv.split(feat_mat):
            # Split the data into train and test sets based on the current fold
            X_train, X_test = (
                feat_mat[train_index],
                feat_mat[test_index],
            )
            y_train, y_test = y[train_index], y[test_index]
            # Fit pipeline model with best hyperparameters
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_preds.append(y_pred)
            y_tests.append(y_test)

            # Calculate evaluation metrics
            #print(y_test, y_pred)
            pearson_r, pval_pearsonr = pearsonr(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            cov_x = np.cov(
                pipeline.named_steps["pca"]
                .transform(X_train)
                .transpose()
                .astype(np.float64)
            )
            cov_y = np.cov(y_train.transpose().astype(np.float64))

            # Append metrics to the respective lists
            pearson_r_scores.append(pearson_r)
            pval_pearsonr_scores.append(pval_pearsonr)
            r2_scores.append(r2)
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            n_components.append(pipeline.named_steps["pca"].n_components_)
            coefficients = pipeline.named_steps["reg"].coef_ # used later to inverse transform and get features coeffs
            pca_ls.append(pipeline.named_steps["pca"])
            pca_coeffs.append(coefficients)

            # debug prints
            # print('feature matrix shape', features_matrix.shape)
            # print('X_test shape', X_test.shape)
            # print('X_test pca transf shape', pipeline.named_steps["pca"].transform(X_test).shape)
            # print('transposed pca tranf shape', pipeline.named_steps["pca"].transform(X_test).transpose().shape)
            # print('cov_x type and shape', type(cov_x), cov_x.shape)
            # print('cov y shape', cov_y.shape)
            # print('coeff transposed shape', coefficients.transpose().shape)
            # correction from Eqn 6 (Haufe et al., 2014)
            corr_coeffs = np.matmul(cov_x, coefficients.transpose()) * (1 / cov_y)
            all_corr_coefficients.append(
                pipeline.named_steps["pca"].inverse_transform(
                    corr_coeffs.transpose()
                )
            )
            all_coefficients.append(
                pipeline.named_steps["pca"].inverse_transform(
                    coefficients.transpose()
                )
            )

        # Permutation tests
        r2score, _, r2p_value = compute_permutation(
            features_matrix,
            y,
            pipeline,
            n_permutations=n_permut,
            cv=cv,
            scoring="r2",
            rdm_seed=rdm_seed
        )
        rmse_score, _, rmse_p_value = compute_permutation(
            features_matrix,
            y,
            pipeline,
            n_permutations= n_permut,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            rdm_seed=rdm_seed
        )
        # Calculate mean metrics across all folds
        mean_pearson_r = np.mean(pearson_r_scores)
        mean_pval_pearsonr = np.mean(pval_pearsonr_scores)
        mean_r2 = np.mean(r2_scores)
        mean_mse = np.mean(mse_scores)
        mean_rmse = np.mean(rmse_scores)
        mean_n_components = np.mean(n_components)

        # print(f"Mean pca comp: p={r2p_value:.4f}")
        print(
            f"Permutation test scores (CV mean) : R2={mean_r2:.4f}, p={r2p_value:.4f} | RMSE={mean_rmse:.4f}, p={rmse_p_value:.4f} | Pearson r={mean_pearson_r:.4f}, p={mean_pval_pearsonr:.4f})"
        )

        # Calculate standard deviation metrics across all folds
        # avg_z_score = np.mean(np.array(all_coefficients), axis=0) / np.std(all_coefficients, axis=0)
        # print(f"Average z-score = {avg_z_score} std = {np.std(all_coefficients, axis=0)}")
        # Plot
        plot_title = f"{pred_metric_name} based CV-prediction of {target_column}"

        

        if doPlot :

            plot_func.reg_plot_performance(
                y_tests,
                y_preds,
                target_column,
                mean_pearson_r,
                mean_rmse,
                mean_r2,
                r2p_value,
                rmse_p_value,
                mean_n_components,
                title=plot_title,
            )

        mean_metrics.append((mean_rmse, mean_mse, mean_r2, mean_pearson_r))

        result_Yi[target_column] = {
            "plot_title": plot_title,
            "CV_mse": mse_scores,
            "CV_rmse": rmse_scores,
            "CV_r2": r2_scores,
            "CV_pearson_r": pearson_r_scores,
            "pca_n_components": n_components,
            "r2_pvals": r2p_value,
            "rmse_pvals": rmse_p_value,
            "pearson_pvals" : pval_pearsonr_scores,
            "y_preds": y_preds,
            "y_tests": y_tests,
            "pca_objects" : pca_ls,
            "pca_reg_coeffs" : pca_coeffs,
            "coeffs" : all_coefficients,
	    "corr_coeffs": all_corr_coefficients,
        }

   # out_dict[metrics_name] = result_Yi

    return result_Yi
