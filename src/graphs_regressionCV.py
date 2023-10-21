import networkx as nx
import numpy as np
import os
import glob as glob
import pickle
import pandas as pd
from sklearn.model_selection import permutation_test_score
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.linear_model import RidgeCV
from statsmodels.stats.multitest import fdrcorrection


def load_process_y(xlsx_path, subjects):
    # dependant variables

    rawY = pd.read_excel(xlsx_path, sheet_name=0, index_col=1, header=2).iloc[
        2:, [4, 17, 18, 19, 38, 48, 65, 67]
    ]
    columns_of_interest = [
        "SHSS_score",
        "raw_change_ANA",
        "raw_change_HYPER",
        "total_chge_pain_hypAna",
        "Chge_hypnotic_depth",
        "Mental_relax_absChange",
        "Automaticity_post_ind",
        "Abs_diff_automaticity",
    ]
    rawY.columns = columns_of_interest
    cleanY = rawY.iloc[:-6, :]  # remove sub04, sub34 and last 6 rows
    cutY = cleanY.drop(["APM04*", "APM34*"])

    filledY = cutY.fillna(cutY.astype(float).mean()).astype(float)
    filledY["SHSS_groups"] = pd.cut(
        filledY["SHSS_score"], bins=[0, 4, 8, 12], labels=["0", "1", "2"]
    )  # encode 3 groups for SHSS scores

    # bin_edges = np.linspace(min(data_column), max(data_column), 4) # 4 bins
    filledY["auto_groups"] = pd.cut(
        filledY["Abs_diff_automaticity"],
        bins=np.linspace(
            min(filledY["Abs_diff_automaticity"]) - 1e-10,
            max(filledY["Abs_diff_automaticity"]) + 1e-10,
            4,
        ),
        labels=["0", "1", "2"],
    )

    # rename 'APM_XX_HH' to 'APMXX' format, for compatibility with Y.rows
    subjects_rewritten = ["APM" + s.split("_")[1] for s in subjects]

    # reorder to match subjects order
    Y = pd.DataFrame(columns=filledY.columns)
    for namei in subjects_rewritten:
        row = filledY.loc[namei]
        Y.loc[namei] = row

    return Y, columns_of_interest


def compute_indiv_graphs_metrics(connectomes, subjects, labels):
    """
    Compute graph metrics for each subject's graph
    and return a list of feature matrices
    results : dict
    Y : dataframe, index = subject names and columns = target columns
    labels : list of strings from the atlas ROIs
    """
    metric_dict = dict(
        nodes=labels,
        degree=[],
        closenessCent=[],
        betweennessCent=[],
        clustering=[],
        communities=[],
    )
    # Initialize single-subject graphs from Adjacency matrix
    As = [connectomes[i] for i in range(len(connectomes))]
    rawGs = {nx.from_numpy_array(A, create_using=nx.Graph) for A in As}
    rawGs = {
        nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels))) for G in rawGs
    }
    # add keys (subjects names) to make it a dict instead od a set
    Gs = dict()
    for name, G in zip(subjects, rawGs):
        Gs[name] = G
    # Adding a strengh attribute to each node
    for participant, graph in Gs.items():
        degree = graph.degree(weight="weight")
        strengths = {node: val for (node, val) in degree}  # Tuple to dict
        nx.set_node_attributes(graph, strengths, "strength")
        norm_strengths = {
            node: val * 1 / (len(graph.nodes) - 1) for (node, val) in degree
        }
        nx.set_node_attributes(graph, norm_strengths, "strengthnorm")
        metric_dict["degree"].append(list(dict(degree).values()))
    # Degree centrality
#    for participant, G in Gs.items():
 #       degree = nx.degree_centrality(G)
  #      nx.set_node_attributes(G, degree, "degCentrality")
   #     metric_dict["degreeCent"].append(list(degree.values()))  # Save metric in a dict

    # betweenness centrality
    for participant, G in Gs.items():
        G_distance_dict = {
            (e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data="weight")
        }  # convert weight to distance to compute betweenness centrality
        nx.set_edge_attributes(G, G_distance_dict, "distance")
        betweenness = nx.betweenness_centrality(G, weight="distance")
        nx.set_node_attributes(G, betweenness, "betCentrality")
        metric_dict["betweennessCent"].append(list(betweenness.values()))
    # closeness centrality
    for participant, G in Gs.items():
        # G_distance_dict = {
        #    (e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data="weight")
        # }
        # nx.set_edge_attributes(G, G_distance_dict, "distance")
        closeness = nx.closeness_centrality(G, distance="distance")
        nx.set_node_attributes(G, closeness, "closecent")
        metric_dict["closenessCent"].append(list(closeness.values()))
    # clustering
    for participant, graph in Gs.items():
        clust = nx.clustering(graph, weight="weight")
        nx.set_node_attributes(graph, clust, "clustering")
        metric_dict["clustering"].append(list(clust.values()))
    # Communities
    for participant, graph in Gs.items():
        communities = nx.community.louvain_communities(
            graph
        )  # Returns a list instead of a dict
        nx.set_node_attributes(graph, communities, "community")
        metric_dict["communities"].append(communities)

    # ---Feature matrices based on subjects' graphs---#
    participant_names = list(Gs.keys())
    node_names = list(Gs[participant_names[0]].nodes())

    # Reorganize data structure from lists of vectors to pd (N x nodes array) for each dict key
    for metric in list(metric_dict.keys()):  # Excluding nodes and communities
        tmp_data = metric_dict[metric]
        if metric != 'communities' and metric != 'nodes':
            metric_dict[metric] = pd.DataFrame(
                np.array(tmp_data), columns=labels, index=subjects
            )

    return metric_dict  # dict w/ keys = metrics and values = list of list vectors (one list per subject)


def metrics_diff_postpre(post_dict, pre_dict, subjects, exclude_keys = []):

    assert post_dict.keys() == pre_dict.keys()
    change_dict = {}

    for metric in list(post_dict.keys()):
        if metric not in exclude_keys:
            postpre_diff = post_dict[metric] - pre_dict[metric]
            df = pd.DataFrame(postpre_diff, columns=post_dict["nodes"], index=subjects)
            change_dict[metric] = df

    return change_dict


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
    )  # store the original con matrix based on the flatten participants' features (shape : N x features)

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
    random_seed=40
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
        cv = kf = KFold(n_splits=5, random_state=random_seed, shuffle=True)
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
        random_state=random_seed,
    )

    return score, perm_scores, pvalue


def regression_cv(graph_dict, Y, target_columns, exclude_keys = [], rdm_seed=40):
    """
    Compute the regression with cross-validation for each graph metric
    and return a dict of results
    X_ls : list of feature matrices
    Y : dataframe, index = subject names and columns = target columns
    target_columns : list of strings from the target columns
    metrics_names : list of strings from the graph metrics but written in a way that will figure in the plots outputs.
    """

    # SVR model for each graph metric
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    from scipy.stats import pearsonr
    import numpy as np

    mean_metrics = []
    out_dict = dict()
    result_per_col = dict()
    random_seed = rdm_seed
    # Output information : dict_keys(['nodes', 'degree', 'closenessCent', 'betweennessCent', 'clustering', 'communities'])
    # ([X_con, X_degree, X_closeness, X_betweenness, X_clustering], ['Connectivity matrix', 'Degree', 'Closeness centrality', 'Betweenness centrality', 'Clustering'])

    for metrics_name in [m for m in list(graph_dict.keys()) if m not in exclude_keys]:
        print("================ \n{}\n================".format(metrics_name))
        # Prep dataframe with std scaler
        rawfeatures_matrix = graph_dict[metrics_name]
        prePipeline = Pipeline([("scaler", StandardScaler())])
        features_matrix = prePipeline.fit_transform(rawfeatures_matrix)
        pipeline = Pipeline([("pca", PCA(n_components=0.80)), ("reg", SVR(kernel='linear'))])
        # cv = KFold(n_splits=5, random_state=random_seed, shuffle=True)
        cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=random_seed)

        for target_column in target_columns:
            print(f"--- {target_column} ---")
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

            for train_index, test_index in cv.split(features_matrix):
                # Split the data into train and test sets based on the current fold
                X_train, X_test = (
                    features_matrix[train_index],
                    features_matrix[test_index],
                )
                y_train, y_test = y[train_index], y[test_index]
                # Fit pipeline model with best hyperparameters
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_preds.append(y_pred)
                y_tests.append(y_test)

                # Calculate evaluation metrics
                pearson_r, pval_pearsonr = pearsonr(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                cov_x = np.cov(
                    pipeline.named_steps["pca"]
                    .transform(X_test)
                    .transpose()
                    .astype(np.float64)
                )
                cov_y = np.cov(y_test.transpose().astype(np.float64))

                # Append metrics to the respective lists
                pearson_r_scores.append(pearson_r)
                pval_pearsonr_scores.append(pval_pearsonr)
                r2_scores.append(r2)
                mse_scores.append(mse)
                rmse_scores.append(rmse)
                n_components.append(pipeline.named_steps["pca"].n_components_)
                coefficients = pipeline.named_steps["reg"].coef_

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
                all_coefficients.append(
                    pipeline.named_steps["pca"].inverse_transform(
                        corr_coeffs.transpose()
                    )
                )

            # Permutation tests
            r2score, _, r2p_value = compute_permutation(
                features_matrix,
                y,
                pipeline,
                n_permutations=5000,
                cv=cv,
                scoring="r2",
                random_seed=random_seed
            )
            rmse_score, _, rmse_p_value = compute_permutation(
                features_matrix,
                y,
                pipeline,
                n_permutations= 5000,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                random_seed=random_seed
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
                f"Permutation test for r2 and RMSE values: p={r2p_value:.4f} and {rmse_p_value:.4f} and Pearson R : {mean_pearson_r:.4f} ({mean_pval_pearsonr:.4f})"
            )

            # Calculate standard deviation metrics across all folds
            # avg_z_score = np.mean(np.array(all_coefficients), axis=0) / np.std(all_coefficients, axis=0)
            # print(f"Average z-score = {avg_z_score} std = {np.std(all_coefficients, axis=0)}")
            # Plot
            plot_title = f"{metrics_name} based CV-prediction of {target_column}"
            # reg_plot_performance(
            #    y_tests,
            #    y_preds,
            #    target_column,
            #    mean_pearson_r,
            #    mean_rmse,
            #    mean_r2,
            #    r2p_value,
            #    rmse_p_value,
            #    mean_n_components,
            #    title=plot_title,
            # )

            mean_metrics.append((mean_rmse, mean_mse, mean_r2, mean_pearson_r))

            result_per_col[target_column] = {
                "plot_title": plot_title,
                "CV_mse": mse_scores,
                "CV_rmse": rmse_scores,
                "CV_r2": r2_scores,
                "CV_pearson_r": pearson_r_scores,
                "pca_n_components": n_components,
                "r2p_value": r2p_value,
                "rmse_p_value": rmse_p_value,
                "y_preds": y_preds,
                "y_tests": y_tests,
                "corr_coeffs": all_coefficients,
            }

        out_dict[metrics_name] = result_per_col
    return out_dict
