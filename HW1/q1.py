import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def standardize_data(trainx, testx):
    """Standardizes features to zero mean, unit variance, using statistics
    computed from the training data only.

    Args:
        trainx: (n_train, p) ndarray of training features.
        testx: (n_test, p) ndarray of test features.

    Returns:
        (train_s, test_s) tuple of standardized ndarrays, same shapes as
        trainx/testx respectively. testx must be standardized using
        trainx's mean/std, not its own.
    """
    mean = np.mean(trainx, axis=0)
    std = np.std(trainx, axis=0)
    # Constant training features cannot have unit variance; avoid dividing by zero.
    std = np.where(std == 0, 1.0, std)
    return (trainx - mean) / std, (testx - mean) / std


def eval_linear(trainx, trainy, testx, testy):
    """Fits an (unregularized) linear regression model on the training data
    and evaluates it on both the training and test data.

    Args:
        trainx: (n_train, p) ndarray of training features.
        trainy: (n_train,) ndarray of training targets.
        testx: (n_test, p) ndarray of test features.
        testy: (n_test,) ndarray of test targets.

    Returns:
        dict with keys "train-rmse", "train-r2", "test-rmse", "test-r2"
        mapping to the corresponding scalar performance metrics.
    """
    model = LinearRegression()
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    test_pred = model.predict(testx)
    return {
        "train-rmse": float(np.sqrt(mean_squared_error(trainy, train_pred))),
        "train-r2": float(r2_score(trainy, train_pred)),
        "test-rmse": float(np.sqrt(mean_squared_error(testy, test_pred))),
        "test-r2": float(r2_score(testy, test_pred)),
    }


def eval_ridge(trainx, trainy, testx, testy, gamma):
    """Fits a ridge regression model on the training data and evaluates it
    on both the training and test data.

    Args:
        trainx: (n_train, p) ndarray of training features.
        trainy: (n_train,) ndarray of training targets.
        testx: (n_test, p) ndarray of test features.
        testy: (n_test,) ndarray of test targets.
        gamma: scalar >= 0, ridge regularization strength (sklearn's alpha).

    Returns:
        dict with keys "train-rmse", "train-r2", "test-rmse", "test-r2"
        mapping to the corresponding scalar performance metrics.
    """
    model = Ridge(alpha=gamma)
    model.fit(trainx, trainy)
    train_pred = model.predict(trainx)
    test_pred = model.predict(testx)
    return {
        "train-rmse": float(np.sqrt(mean_squared_error(trainy, train_pred))),
        "train-r2": float(r2_score(trainy, train_pred)),
        "test-rmse": float(np.sqrt(mean_squared_error(testy, test_pred))),
        "test-r2": float(r2_score(testy, test_pred)),
    }


def ridge_sweep(trainx, trainy, testx, testy, gammas, save_coefs=False):
    rows = []
    coefs = []

    for gamma in gammas:
        model = Ridge(alpha=gamma)
        model.fit(trainx, trainy)

        train_pred = model.predict(trainx)
        test_pred = model.predict(testx)

        rows.append({
            "gamma": gamma,
            "train-rmse": float(np.sqrt(mean_squared_error(trainy, train_pred))),
            "train-r2": float(r2_score(trainy, train_pred)),
            "test-rmse": float(np.sqrt(mean_squared_error(testy, test_pred))),
            "test-r2": float(r2_score(testy, test_pred)),
        })

        if save_coefs:
            coefs.append(model.coef_.copy())

    results = pd.DataFrame(rows)

    if save_coefs:
        return results, np.array(coefs)

    return results


# def print_ridge_results(label, results):
#     """Print the sweep and return the row with the lowest test RMSE."""
#     print(f"\nQ1e: {label} data")
#     formatters = {col: "{:.6f}".format for col in results.columns}
#     formatters["gamma"] = "{:.8e}".format
#     print(results.to_string(index=False, formatters=formatters))
#     best = results.loc[results["test-rmse"].idxmin()]
#     print(
#         f"Selected gamma: {best['gamma']:.8e} (lowest test RMSE on this grid)\n"
#         f"Train RMSE: {best['train-rmse']:.6f}, R^2: {best['train-r2']:.6f}\n"
#         f"Test  RMSE: {best['test-rmse']:.6f}, R^2: {best['test-r2']:.6f}"
#     )
#     return best
def print_ridge_results(label, results):
    """
    Print ridge results and return the row with the lowest test RMSE.
    """
    print(f"\n{label} data")

    formatters = {
        "gamma": "{:.8e}".format,
        "train-rmse": "{:.6f}".format,
        "train-r2": "{:.6f}".format,
        "test-rmse": "{:.6f}".format,
        "test-r2": "{:.6f}".format,
    }

    print(results.to_string(index=False, formatters=formatters))

    best = results.loc[results["test-rmse"].idxmin()]

    print(f"\nSelected gamma: {best['gamma']:.8e}")
    print(
        f"Train RMSE: {best['train-rmse']:.6f}, "
        f"R^2: {best['train-r2']:.6f}"
    )
    print(
        f"Test RMSE: {best['test-rmse']:.6f}, "
        f"R^2: {best['test-r2']:.6f}"
    )

    return best

def fit_station_categories(train_df, min_count=5):
    """Learn station categories from training rows, pooling rare stations."""
    # handle unseen stations
    categories = {}
    for col in ["start_station_id", "end_station_id"]:
        stations = train_df[col].astype("string")
        counts = stations.value_counts()
        kept = counts[counts >= min_count].index.tolist()
        # Give the fallback training examples even when no stations are rare.
        if kept and len(kept) == len(counts) and not stations.isna().any():
            kept.remove(counts.idxmin())
        categories[col] = ["__other__"] + sorted(kept)
    return categories


def preprocess_features(df, use_weekend=False, station_categories=None):
    """Applies the README transformations and preserves all other columns.

    Drops ride_id and station names, replaces station IDs and started_at with encoded
    features, and keeps one integer indicator per bike/membership pair.
    All other columns, including distance features and duration_min,
    are retained unchanged.

    Args:
        df: pandas DataFrame of processed CitiBike trip data.
        use_weekend: If True, use a weekend indicator instead of the seven
            day-of-week encoding. Defaults to False.
        station_categories: Training vocabulary from fit_station_categories.
            If omitted, learn it from df. Pass the same vocabulary for test data.

    Returns:
        pandas DataFrame containing the transformed features alongside
        every untouched input column. The input DataFrame is not modified.
    """
    df = df.copy()
    # handle unseen stations
    # Rare (fewer than five training rides), missing, and unseen stations share
    # the training-supported __other__ category. Never learn categories on test.
    if station_categories is None:
        station_categories = fit_station_categories(df)
    for col, categories in station_categories.items():
        stations = df[col].astype("string")
        df[col] = pd.Categorical(
            stations.where(stations.isin(categories[1:]), "__other__"),
            categories=categories,
        )
    started_at = pd.to_datetime(df["started_at"])

    # Fix temporal categories so even absent hours/months get dummy columns.
    df["hour"] = pd.Categorical(started_at.dt.hour, categories=range(24))
    df["month"] = pd.Categorical(started_at.dt.month, categories=range(1, 13))
    categorical_cols = ["start_station_id", "end_station_id", "hour", "month"]
    if use_weekend:
        df["weekend"] = (started_at.dt.dayofweek >= 5).astype(int)
    else:
        # Monday is 0 and Sunday is 6.
        df["day_of_week"] = pd.Categorical(
            started_at.dt.dayofweek, categories=range(7)
        )
        categorical_cols.append("day_of_week")

    binary_cols = [
        "rideable_type_classic_bike",
        "rideable_type_electric_bike",
        "member_casual_member",
        "member_casual_casual",
    ]
    df[binary_cols] = df[binary_cols].astype(int)
    #remove dummy variable trap
    # Omit the reference category: hour 0, January, Monday, and station __other__.
    # Electric bikes and casual riders are the references for the binary pairs.
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    df = df.drop(columns=["rideable_type_electric_bike", "member_casual_casual"])
    return df.drop(
        columns=["ride_id", "started_at", "start_station_name", "end_station_name"],
        errors="ignore",
    )

def graph_correlations(DF):
    """
    graph pairwise correlations to get a sense of which features might work well in a model
    """
    features = [
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        "duration_min",
        "distance_km",
        "manhattan_km",
        "distance_km_sq",
        "distance_km_log1p",
        "rideable_type_classic_bike",
        "rideable_type_electric_bike",
        "member_casual_member",
        "member_casual_casual"
    ]

    # Calculate Pearson correlation between every pair of features
    corr = DF[features].corr()

    # # Hide the upper triangle because the correlation matrix is symmetric
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create figure
    plt.figure(figsize=(14, 10))

    # Draw correlation heatmap
    sns.heatmap(
        corr,
        # mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


def main():
    """Example driver showing how to read the CitiBike CSVs, engineer
    features, and evaluate a model. This is a starting point for all the
    written analysis part of the problem. You can extend it in main or
    use a different file to call the functions.

    Assumes the data sits in the current working directory.
    """
    train_df = pd.read_csv("citibike_2022_train_processed.csv", low_memory=False)
    test_df = pd.read_csv("citibike_2023_test_processed.csv", low_memory=False)

    # graph_correlations(train_df) 

    # handle unseen stations
    station_categories = fit_station_categories(train_df)
    train_df = preprocess_features(train_df, station_categories=station_categories)
    test_df = preprocess_features(test_df, station_categories=station_categories)
    # Keep the training station vocabulary and column order for prediction.
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    feature_cols = [c for c in train_df.columns if c not in ("duration_min", "ride_id")]
    train_y = train_df["duration_min"].to_numpy()
    test_y = test_df["duration_min"].to_numpy()
    train_x = train_df[feature_cols].to_numpy(dtype=float)
    test_x = test_df[feature_cols].to_numpy(dtype=float)

    train_x_std, test_x_std = standardize_data(train_x, test_x)

    # TODO: Do all of the written parts of the problem below.

    # Q1d. run eval_linear here
    print("printing linear regression results. Q1d")
    results = eval_linear(train_x_std, train_y, test_x_std, test_y)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Q1e. run ridge here
    print("printing ridge results. Q1e &f")
    gammas = np.logspace(-4, 10, 20)
    standardized_results, standardized_coefs = ridge_sweep(
        train_x_std, train_y, test_x_std, test_y,
        gammas,
        save_coefs=True
    ) #saved for Q1f
    original_results = ridge_sweep(
        train_x, train_y, test_x, test_y,
        gammas
    )
    best_std = print_ridge_results("Standardized", standardized_results)
    best_original = print_ridge_results("Unstandardized", original_results)

    print("\nQ1e: Comparison of the best models on the tested grid")
    changed = best_std["gamma"] != best_original["gamma"]
    print(f"Does standardization change the selected gamma? {'Yes' if changed else 'No'}")
    rmse_difference = best_std["test-rmse"] - best_original["test-rmse"]
    r2_difference = best_std["test-r2"] - best_original["test-r2"]
    print(f"Test RMSE difference (standardized - unstandardized): {rmse_difference:+.6f}")
    print(f"Test R^2 difference (standardized - unstandardized): {r2_difference:+.6f}")
    if rmse_difference < 0:
        print("Standardization improves the best test performance on this grid.")
    elif rmse_difference > 0:
        print("Standardization worsens the best test performance on this grid.")
    else:
        print("The best test performance is equal on this grid.")
    print("Lower RMSE and higher R^2 are better; both select the same minimum-error model.\n")


    # Q1f
    print("\nQ1f: Ridge regression")

    # 20 log-spaced gamma values from 10^-4 to 10^10
    gammas = np.logspace(-4, 10, 20)

    # Standardized data.
    # Save coefficients here because Q1h needs these exact same fitted models.
    standardized_results, standardized_coefs = ridge_sweep(
        train_x_std,
        train_y,
        test_x_std,
        test_y,
        gammas,
        save_coefs=True
    )

    # Unstandardized data.
    # Q1h does not need these coefficients, so don't save them.
    unstandardized_results = ridge_sweep(
        train_x,
        train_y,
        test_x,
        test_y,
        gammas
    )

    best_std = print_ridge_results(
        "Standardized",
        standardized_results
    )

    best_unstd = print_ridge_results(
        "Unstandardized",
        unstandardized_results
    )

    print("\nComparison")

    print(
        f"Best standardized gamma: "
        f"{best_std['gamma']:.8e}"
    )

    print(
        f"Best unstandardized gamma: "
        f"{best_unstd['gamma']:.8e}"
    )

    print(
        f"Test RMSE difference (standardized - unstandardized): "
        f"{best_std['test-rmse'] - best_unstd['test-rmse']:+.6f}"
    )

    print(
        f"Test R^2 difference (standardized - unstandardized): "
        f"{best_std['test-r2'] - best_unstd['test-r2']:+.6f}"
    )


if __name__ == "__main__":
    main()
