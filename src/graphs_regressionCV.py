import networkx as nx
import numpy as np
import os
import glob as glob
import pickle
import func
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
    rawY.columns = [
        "SHSS_score",
        "raw_change_ANA",
        "raw_change_HYPER",
        "total_chge_pain_hypAna",
        "Chge_hypnotic_depth",
        "Mental_relax_absChange",
        "Automaticity_post_ind",
        "Abs_diff_automaticity",
    ]

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

    # reorder to match order on elm server e.i. ENV of server used to run analyses!
    new_order = [
        "APM01",
        "APM16",
        "APM06",
        "APM38",
        "APM12",
        "APM03",
        "APM07",
        "APM28",
        "APM29",
        "APM17",
        "APM11",
        "APM02",
        "APM15",
        "APM05",
        "APM32",
        "APM42",
        "APM35",
        "APM43",
        "APM41",
        "APM08",
        "APM36",
        "APM27",
        "APM33",
        "APM22",
        "APM20",
        "APM09",
        "APM37",
        "APM26",
        "APM47",
        "APM46",
        "APM40",
    ]
    if new_order != subjects:
        new_order = subjects

    # reorder to match elm server order
    Y = pd.DataFrame(columns=filledY.columns)
    for name in new_order:
        row = filledY.loc[name]
        Y.loc[name] = row
    # List of target columns for prediction
    target_columns = [
        "SHSS_score",
        "raw_change_ANA",
        "raw_change_HYPER",
        "total_chge_pain_hypAna",
        "Chge_hypnotic_depth",
        "Mental_relax_absChange",
        "Automaticity_post_ind",
        "Abs_diff_automaticity",
    ]

    return Y, target_columns


def graph_metrics(results, Y, labels):
    """
    Compute graph metrics for each subject's graph
    and return a list of feature matrices
    results : dict
    Y : dataframe, index = subject names and columns = target columns
    labels : list of strings from the atlas ROIs
    """
    # Single-subject graphs
    As = [
        results["contrast_connectomes"][i]
        for i in range(len(results["contrast_connectomes"]))
    ]
    rawGs = {nx.from_numpy_array(A, create_using=nx.Graph) for A in As}
    rawGs = {
        nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels))) for G in rawGs
    }

    # add keys to make it a dict instead od a set
    Gs = dict()
    for name, G in zip(Y.index, rawGs):
        Gs[name] = G
        # Gs[].edges(data=True)

    for participant, graph in Gs.items():
        strength = graph.degree(weight="weight")
        strengths = {node: val for (node, val) in strength}
        nx.set_node_attributes(graph, strengths, "strength")
        norm_strengths = {
            node: val * 1 / (len(graph.nodes) - 1) for (node, val) in strength
        }
        nx.set_node_attributes(graph, norm_strengths, "strengthnorm")

    # Graph metric for each subjects' graph
    # Degree centrality
    for participant, G in Gs.items():
        degree = nx.degree_centrality(G)
        nx.set_node_attributes(G, degree, "degCentrality")
    # betweenness centrality
    for participant, G in Gs.items():
        G_distance_dict = {
            (e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data="weight")
        }
        nx.set_edge_attributes(G, G_distance_dict, "distance")
        closeness = nx.betweenness_centrality(G, weight="distance")
        nx.set_node_attributes(G, closeness, "betCentrality")
    # closeness centrality
    for participant, G in Gs.items():
        G_distance_dict = {
            (e1, e2): 1 / abs(weight) for e1, e2, weight in G.edges(data="weight")
        }
        nx.set_edge_attributes(G, G_distance_dict, "distance")
        closeness = nx.closeness_centrality(G, distance="distance")
        nx.set_node_attributes(G, closeness, "closecent")
    for participant, graph in Gs.items():
        clust = nx.clustering(graph, weight="weight")
        nx.set_node_attributes(graph, clust, "clustering")

    # ---Feature matrices based on subjects' graphs---#
    participant_names = list(Gs.keys())
    node_names = list(Gs[participant_names[0]].nodes())
    # Adjacency matrix as a feature matrix see shape in prints
    X_con = results["contrastX"]

    X_degree = np.zeros((len(participant_names), len(node_names)))
    # Fill the features matrix with degree strength values
    for i, participant in enumerate(participant_names):
        graph = Gs[participant]
        degrees = nx.get_node_attributes(graph, "strengthnorm")
        for j, node in enumerate(node_names):
            X_degree[i, j] = degrees[node]

    X_degreeCentrality = np.zeros((len(participant_names), len(node_names)))
    for i, participant in enumerate(participant_names):
        graph = Gs[participant]
        degreeCentrality = nx.get_node_attributes(graph, "degCentrality")
        for j, node in enumerate(node_names):
            X_degreeCentrality[i, j] = degreeCentrality[node]

    X_closeness = np.zeros((len(participant_names), len(node_names)))
    for i, participant in enumerate(participant_names):
        graph = Gs[participant]
        closeness = nx.get_node_attributes(graph, "closecent")
        for j, node in enumerate(node_names):
            X_closeness[i, j] = closeness[node]

    X_betweenness = np.zeros((len(participant_names), len(node_names)))
    for i, participant in enumerate(participant_names):
        graph = Gs[participant]
        betweenness = nx.get_node_attributes(graph, "betCentrality")
        for j, node in enumerate(node_names):
            X_betweenness[i, j] = betweenness[node]

    X_clustering = np.zeros((len(participant_names), len(node_names)))
    for i, participant in enumerate(participant_names):
        graph = Gs[participant]
        clustering = nx.get_node_attributes(graph, "clustering")
        for j, node in enumerate(node_names):
            X_clustering[i, j] = clustering[node]

    X_ls = [
        X_con,
        X_degree,
        X_degreeCentrality,
        X_closeness,
        X_betweenness,
        X_clustering,
    ]
    metrics = [
        "connectivity_matrix",
        "Degree",
        "closeness",
        "betweenness",
        "clustering",
    ]

    return X_ls, metrics


def compute_permutation(
    X,
    y,
    pipeline,
    gr=None,
    cv=None,
    n_components=0.80,
    n_permutations=5000,
    scoring="r2",
    random_seed=40,
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


def regression_cv(X_ls, Y, target_columns, metrics_names):
    """
    Compute the regression with cross-validation for each graph metric
    and return a dict of results
    X_ls : list of feature matrices
    Y : dataframe, index = subject names and columns = target columns
    target_columns : list of strings from the target columns
    metrics_names : list of strings from the graph metrics
    """

    # SVR model for each graph metric
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from scipy.stats import pearsonr
    import numpy as np

    mean_metrics = []
    result_dict = dict()
    result_per_col = dict()
    random_seed = 0
    # Example of X_ls and metrics_names :
    # ([X_con, X_degree, X_closeness, X_betweenness, X_clustering], ['Connectivity matrix', 'Degree', 'Closeness centrality', 'Betweenness centrality', 'Clustering'])
    for rawfeatures_matrix, g_metric in zip(X_ls, metrics_names):
        print("================ \n{}\n================".format(g_metric))
        prePipeline = Pipeline([("scaler", StandardScaler())])
        features_matrix = prePipeline.fit_transform(rawfeatures_matrix)
        pipeline = Pipeline(
            [("pca", PCA(n_components=0.90)), ("reg", SVR(kernel="linear"))]
        )
        # cv = KFold(n_splits=5, random_state=random_seed, shuffle=True)
        cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=random_seed)

        for target_column in target_columns:
            print(f"--- {target_column} ---")
            y_preds = []
            y_tests = []
            y = Y[target_column].values

            pearson_r_scores = []
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
                pearson_r, _ = pearsonr(y_test, y_pred)
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
                r2_scores.append(r2)
                mse_scores.append(mse)
                rmse_scores.append(rmse)
                n_components.append(pipeline.named_steps["pca"].n_components_)
                coefficients = pipeline.named_steps["reg"].coef_

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
                cv=cv,
                scoring="r2",
                random_seed=random_seed,
            )
            rmse_score, _, rmse_p_value = compute_permutation(
                features_matrix,
                y,
                pipeline,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                random_seed=random_seed,
            )

            print(f"Permutation test for r2 values: p={r2p_value:.4f}")
            # Calculate mean metrics across all folds
            mean_pearson_r = np.mean(pearson_r_scores)
            mean_r2 = np.mean(r2_scores)
            mean_mse = np.mean(mse_scores)
            mean_rmse = np.mean(rmse_scores)
            mean_n_components = np.mean(n_components)
            # Calculate standard deviation metrics across all folds
            # avg_z_score = np.mean(np.array(all_coefficients), axis=0) / np.std(all_coefficients, axis=0)
            # print(f"Average z-score = {avg_z_score} std = {np.std(all_coefficients, axis=0)}")
            # Plot
            plot_title = f"{g_metric} based CV-prediction of {target_column}"
            reg_plot_performance(
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

            mean_metrics.append((mean_mse, mean_mae, mean_r2, mean_pearson_r))

            result_per_col[target_column] = {
                "plot_title": plot_title,
                "mean_mse": mean_mse,
                "mean_rmse": mean_rmse,
                "mean_r2": mean_r2,
                "mean_pearson_r": mean_pearson_r,
                "mean_n_components": mean_n_components,
                "r2p_value": r2p_value,
                "rmse_p_value": rmse_p_value,
                "y_preds": y_preds,
                "y_tests": y_tests,
                "corr_coeffs": all_coefficients,
            }

        result_dict[g_metric] = result_per_col
    return result_dict
