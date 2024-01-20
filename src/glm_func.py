import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sm
import statsmodels.api as sms
import scipy.stats.stats
import nilearn.image as image
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_epi, plot_roi, show


# function checks column outliers and skew; tests normality showing a histogram
def check_normality(df, col, bins=20):
    """Plots histogram; prints 4th positive standard deviation and observation
    counts, skew and kurtosis for column normality assessment.

    Parameters:
    df (pd.DataFrame): Name of Pandas dataframe.
    col (string): Name of the column to check; must be in df.columns.
    bins (int): Number of bins in histogram. Default set to 20.

    Returns: No return
    """

    std4 = df[col].mean() + 4 * df[col].std()
    std4_cnt = len(df[df[col] > std4])
    print(
        f"{col} 4 Std Dev: {std4} | obs. above this size: {std4_cnt} | Skew: {df[col].skew()} | Kurtosis: {df[col].kurtosis()}"
    )
    df[col].hist(figsize=(8, 4), bins=bins)
    #plt.show()

    return


# define function to generate Top N values, counts and % total for a column
def topn_count(df, column, topn):
    c = df[column].value_counts(dropna=False)
    p = (
        df[column]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )
    cp = (
        100 * df.groupby(column).size().cumsum() / df.groupby(column).size().sum()
    ).round(1).astype(str) + "%"
    print(f"Top 10 Counts By {column.title()}")
    return pd.concat([c, p, cp], axis=1, keys=["Counts", "%", "Cum %"]).iloc[:topn]


def resid_vi(xi, resid, name):
    sns.set_style("darkgrid")
    sns.mpl.rcParams["figure.figsize"] = (15.0, 9.0)
    fig, ax = plt.subplots(1, 1)
    f = sns.regplot(x=xi, y=resid, lowess=True, line_kws={"color": "red"})
    f.set_title("Residuals vs {}".format(name), fontsize=16)
    f.set(xlabel=name, ylabel="Residuals")


# define function to generate 3 plots for X and Y columns in a dataframe: Histogram, Price Box Plot and Top N % Distribution
def distplots(df, xcol, ycol, topn):
    # Set a figure with 3 subplots and figure-level settings
    f, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6))
    sns.set_theme(style="ticks", palette="deep")
    sns.set_style(style="whitegrid")
    f.suptitle("Column Value Distributions", fontsize=14, fontweight="bold")

    # Subplot 1 - Histogram by desired xcol showing probability %
    ax = plt.subplot2grid((1, 7), (0, 0), colspan=2)
    ax = sns.histplot(
        x=df[xcol], color="skyblue", stat="probability", discrete=True, ax=ax
    )
    ax.set_title("Histogram")

    # Subplot 2 - Scatter plot xcol vs. ycol
    ax2 = plt.subplot2grid((1, 7), (0, 2), colspan=2)
    ax2 = sns.scatterplot(x=df[xcol], y=df[ycol])
    ax2.set_title(f"{ycol.title()} vs {xcol.title()} Scatterplot")

    # Subplot 3 - Boxplot by desired xcol against ycol (e.g. price) to see distributions grouped by xcol
    ax3 = plt.subplot2grid((1, 7), (0, 4), colspan=2)
    ax3 = sns.boxplot(
        x=df[xcol], y=df[ycol], showfliers=False, color="skyblue", ax=ax3
    )  # Excludes outliers for presentability
    ax3.set_title("Boxplot")

    # Calculate medians and number of observations per group for use in positioning labels on plot
    medians = df.groupby([xcol])[ycol].median().values
    nobs = df[xcol].value_counts(sort=False).sort_index().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]

    # Add number of observations to the boxplot for indication of each box's relative likelihood
    pos = range(len(nobs))
    for tick, label in zip(pos, ax2.get_xticklabels()):
        ax3.text(
            pos[tick],
            medians[tick] + 0.2,
            nobs[tick],
            horizontalalignment="center",
            size="x-small",
            color="w",
            weight="bold",
        )

    # Subplot 4 - Add dataframe top N value counts and % of group as a table in 3rd plot
    top = topn_count(df, xcol, topn)
    ax4 = plt.subplot2grid((1, 7), (0, 6))
    ax4.table(
        cellText=top.values,
        rowLabels=top.index,
        colLabels=top.columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    ax4.axis("off")
    ax4.set_title("Top Values")

    f.tight_layout()
    plt.show()


# linearity test
def linearity_test(model, y):
    """
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.

    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    """
    sns.set_style("darkgrid")
    sns.mpl.rcParams["figure.figsize"] = (15.0, 9.0)

    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1, 2)
    # ax.update(wspace=0.5, hspace=0.5)
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={"color": "red"})
    ax[0].set_title("Observed vs. Predicted Values", fontsize=16)
    ax[0].set(xlabel="Predicted", ylabel="Observed")

    sns.regplot(
        x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={"color": "red"}
    )
    ax[1].set_title("Residuals vs. Predicted Values", fontsize=16)
    ax[1].set(xlabel="Predicted", ylabel="Residuals")


def plot_Xi_resid(xi, resids):
    """
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.

    Args:
    * model - fitted OLS model from statsmodels
    """
    sns.set_style("darkgrid")
    sns.mpl.rcParams["figure.figsize"] = (15.0, 9.0)

    fig, ax = plt.subplots(1, 1)

    f = sns.regplot(x=xi, y=resids, lowess=True, ax=ax[0], line_kws={"color": "red"})
    f.set_title("Residuals vs Fitted", fontsize=16)
    f.set(xlabel="Fitted Values", ylabel="Residuals")


def normality_of_residuals_test(model):
    """
    Function for drawing the normal QQ-plot of the residuals and running 4 statistical tests to
    investigate the normality of residuals.

    Arg:
    * model - fitted OLS models from statsmodels
    """
    fig = sms.ProbPlot(model.resid).qqplot(line="s")
    plt.title("Q-Q plot")

    jb = stats.jarque_bera(model.resid)
    sw = stats.shapiro(model.resid)
    ad = stats.anderson(model.resid, dist="norm")
    ks = stats.kstest(model.resid, "norm")

    print(f"Jarque-Bera test ---- statistic: {jb[0]:.4f}, p-value: {jb[1]}")
    print(f"Shapiro-Wilk test ---- statistic: {sw[0]:.4f}, p-value: {sw[1]:.4f}")
    print(
        f"Kolmogorov-Smirnov test ---- statistic: {ks.statistic:.4f}, p-value: {ks.pvalue:.4f}"
    )
    print(
        f"Anderson-Darling test ---- statistic: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}"
    )
    print(
        "If the returned AD statistic is larger than the critical value, then for the 5% significance level, the null hypothesis that the data come from the Normal distribution should be rejected. "
    )


def plot_cooks_distance(c, threshold):
    _, ax = plt.subplots(figsize=(9, 6))
    ax.bar
    ax.stem(c, markerfmt=",")
    ax.set_xlabel("instance")
    ax.set_ylabel("distance")
    ax.set_title("Cook's Distance Outlier Detection")
    plt.axhline(threshold, color="red", ls="dotted")
    plt.show()
    return ax


def winzo(array):
    array = x_constant
    # Creating outliers
    # Here, the values which are selected for creating outliers
    # are appended so that same outliers are not created again.
    AlreadySelected = []
    i = 0

    # Creating 5 outliers on the lower end
    while i < 5:
        x = np.random.choice(array)  # Randomly selecting a value from the array
        y = x - mean * 3
        array = np.append(array, y)
        if x not in already_selected:
            AlreadySelected.append(y)

            i += 1

        else:
            continue

    # Creating 5 outliers on the upper end
    i = 0
    while i < 5:
        x = np.random.choice(array)  # Randomly selecting a value from the array
        y = x + mean * 4
        array = np.append(array, y)
        if x not in already_selected:
            AlreadySelected.append(y)

            i += 1

        else:
            continue

    std = np.std(array)  # Storing the standard deviation of the array
    mean = np.mean(array)  # Storing the mean of the array

    plt.boxplot(array)
    plt.title("Array with Outliers")
    plt.show()
