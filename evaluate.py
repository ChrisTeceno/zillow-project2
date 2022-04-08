import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import warnings
import seaborn as sns
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
from matplotlib import style

style.use("ggplot")


def plot_residuals(y, yhat):
    """plot residuals given y and yhat"""
    residuals = y - yhat
    plt.hlines(0, y.min(), y.max(), ls="--")
    plt.scatter(y, residuals, color="blue")
    plt.ylabel("residual ($y - \hat{y}$)")
    plt.xlabel("y value ($y$)")
    plt.title("Actual vs Residual")
    plt.show()


def regression_errors(y, yhat):
    """return metrics"""
    residuals = y - yhat
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "ESS": ((yhat - y.mean()) ** 2).sum(),
            "TSS": ((y - yhat.mean()) ** 2).sum(),
            "MSE": mean_squared_error(y, yhat),
            "RMSE": mean_squared_error(y, yhat) ** 0.5,
        }
    )


def baseline_mean_errors(y):
    """return baseline metrics"""
    # make a series of the baseline value
    mean = pd.Series([y.mean()])
    # repeat the value to make a correctly sized series to match y
    mean = mean.repeat(len(y))
    residuals = y - mean
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "MSE": mean_squared_error(y, mean),
            "RMSE": mean_squared_error(y, mean) ** 0.5,
        }
    )


def baseline_median_errors(y):
    """return baseline metrics"""
    # make a series of the baseline value
    median = pd.Series([y.median()])
    # repeat the value to make a correctly sized series to match y
    median = median.repeat(len(y))
    residuals = y - median
    return pd.Series(
        {
            "SSE": (residuals ** 2).sum(),
            "MSE": mean_squared_error(y, median),
            "RMSE": mean_squared_error(y, median) ** 0.5,
        }
    )


def better_than_baseline(y, yhat):
    """compare model results to baseline based on mean"""
    # make a series of the baseline value
    mean = pd.Series([y.mean()])
    # repeat the value to make a correctly sized series to match y
    mean = mean.repeat(len(y))
    rmse_baseline = (mean_squared_error(y, mean) ** 0.5,)
    rmse_model = (mean_squared_error(y, yhat) ** 0.5,)
    is_better = rmse_model < rmse_baseline
    # print result
    print(f"based on RMSE, is the model better: {is_better}")
    # return a boolean to be used in a df
    return is_better


def histograms_of_data(df):
    """this function will look at the distribution of the target and a few other variables"""
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 8)
    # pick the columns to plot
    cols = [
        "taxvaluedollarcnt",
        "bedroomcnt",
        "bathroomcnt",
        "calculatedfinishedsquarefeet",
        "age",
        "logerror",
    ]
    # run throught the columns and plot the distribution
    for i, col in enumerate(cols):
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1
        # Create subplot.
        # plt.subplot(row X col, where?)
        plt.subplot(1, len(cols), plot_number)
        # Title with column name.
        plt.title(col)
        # Display histogram for column.
        df[col].hist(bins=20)


def graph_variables_vs_target(df, target="logerror"):
    """This function will graph the variables vs the target"""
    # lets look at the relationship between the target variable and the other variables
    # lets change figure size
    plt.rcParams["figure.figsize"] = (20, 10)
    # pick the columns to plot
    cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    # number of cols
    ncols = len(cols)
    # plt.subplot(row X col, where?)
    fig, axes = plt.subplots(int(ncols / 4), 4, sharey=True)
    # run throught the columns and plot the distribution
    for i, col in enumerate(cols):
        x = i % 4
        y = int(i / 4)
        # Title with column name.
        axes[y, x].set_title(col)
        # Display lmplot for column.
        sns.regplot(
            data=df, x=col, y=target, line_kws={"color": "blue"}, ax=axes[y, x],
        )


def elbow_graph(df):
    """make and elbow graph to determine a good k"""
    with plt.style.context("seaborn-whitegrid"):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(df).inertia_ for k in range(2, 12)}).plot(
            marker="x"
        )
        plt.xticks(range(2, 12))
        plt.xlabel("k")
        plt.ylabel("inertia")
        plt.title("Change in inertia as k increases")


def cluster(
    df, k, cluster_num, X_train_scaled, X_validate_scaled, X_test_scaled,
):
    """cluster the data using kmeans"""
    columns = df.columns.to_list()
    cluster_num = "cluster_num" + str(cluster_num)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    kmeans.predict(df)
    cols = df.columns.to_list()
    X_train_scaled[cluster_num] = kmeans.predict(df)
    X_train_scaled[cluster_num] = X_train_scaled[cluster_num].astype("category")
    X_validate_scaled[cluster_num] = kmeans.predict(X_validate_scaled[columns])
    X_test_scaled[cluster_num] = kmeans.predict(X_test_scaled[columns])
    rel = sns.relplot(
        x=cols[0], y=cols[1], hue=cluster_num, data=X_train_scaled, col="logerror_bins"
    )
    rel.fig.suptitle("scaled with k = {}".format(k))


def run_model(
    features,
    model_number,
    X_train_scaled,
    X_validate_scaled,
    y_train,
    y_validate,
    baseline_train,
):
    """run a model and return the rmse"""
    # make and fit the model
    model = LinearRegression().fit(X_train_scaled[features], y_train)
    # make predictions
    y_train_pred = model.predict(X_train_scaled[features])
    y_validate_pred = model.predict(X_validate_scaled[features])
    # compute the RMSE
    rmse_train = mean_squared_error(y_train, y_train_pred) ** (1 / 2)
    rmse_validate = mean_squared_error(y_validate, y_validate_pred) ** (1 / 2)
    # add the rmse to the results dataframe
    model_results = pd.Series(
        {
            "model_number": model_number,
            "model_type": "Linear Regression",
            "features": features,
            "RMSE_train": rmse_train,
            "RMSE_validate": rmse_validate,
            "baseline": baseline_train,
            "better_than_baseline": rmse_train < baseline_train,
        }
    )
    # return the results
    return model_results


def select_RFE(X, y, k=2):
    """pick k features with the highest RFE score"""
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    # return pd.DataFrame({'rfe_ranking': rfe.ranking_}, index=X_train.columns)
    return X.columns[rfe.get_support()]


def make_rfe_ranking(X_train, y_train, n=3):
    """make the ranking for the rfe"""
    # make a model for RFE
    model = LinearRegression()
    # use recursive feature elimination to select features
    rfe = RFE(model, n_features_to_select=n)
    # fit the RFE to the training with only the original and not the scaled features
    rfe.fit(X_train, y_train)
    return pd.DataFrame({"rfe_ranking": rfe.ranking_}, index=X_train.columns,)


def run_on_test(
    features,
    X_train_scaled,
    X_validate_scaled,
    X_test_scaled,
    y_train,
    y_validate,
    y_test,
    baseline_train,
):
    """run the model on the test data"""
    # rebuild the best model
    model = LinearRegression()
    # fit on the combination determined above
    model.fit(X_train_scaled[features], y_train)
    # make predictions
    y_train_pred = model.predict(X_train_scaled[features])
    y_validate_pred = model.predict(X_validate_scaled[features])
    y_test_pred = model.predict(X_test_scaled[features])
    # compute the RMSE
    rmse_train = mean_squared_error(y_train, y_train_pred) ** (1 / 2)
    rmse_validate = mean_squared_error(y_validate, y_validate_pred) ** (1 / 2)
    rmse_test = mean_squared_error(y_test, y_test_pred) ** (1 / 2)
    # build df
    stats = pd.Series(
        {
            "model_number": 46,
            "model_type": "LinearRegression",
            "number_of_features": len(features),
            "RMSE_train": rmse_train,
            "RMSE_validate": rmse_validate,
            "RMSE_test": rmse_test,
            "baseline": baseline_train,
            "better_than_baseline": rmse_train < baseline_train,
        }
    )

    # show the results of the best model
    return stats
