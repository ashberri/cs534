import argparse
import json
from pathlib import Path
from time import perf_counter

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


def ridge_sweep(trainx, trainy, testx, testy, gammas, save_coefs=False,
                progress=None, label="Ridge"):
    rows = []
    coefs = []

    for i, gamma in enumerate(gammas, start=1):
        started = perf_counter()
        if progress:
            progress(f"{label}: gamma {i}/{len(gammas)} ({gamma:.2e}) — fitting")
        model = Ridge(alpha=gamma)
        model.fit(trainx, trainy)

        if progress:
            progress(f"{label}: gamma {i}/{len(gammas)} — predicting and scoring")
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
        if progress:
            progress(
                f"{label}: gamma {i}/{len(gammas)} complete in "
                f"{perf_counter() - started:.1f}s; test RMSE={rows[-1]['test-rmse']:.6f}"
            )

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


SAVED_RESULTS_PATH = Path(__file__).resolve().with_name("saved_coef.txt")


def save_results(path, results, standardized_results, unstandardized_results,
                 standardized_coefs, gammas, feature_cols, station_categories,
                 train_mean, train_std):
    """Save report/plot inputs and preprocessing metadata, without the datasets."""
    best_idx = standardized_results["test-rmse"].idxmin()
    best_std = standardized_results.loc[best_idx]
    best_unstd = unstandardized_results.loc[unstandardized_results["test-rmse"].idxmin()]
    best_coefs = standardized_coefs[best_idx]
    top_idx = np.argsort(np.abs(best_coefs))[-3:][::-1]
    saved = {
        "format_version": 1,
        "linear_results": results,
        "standardized_results": standardized_results.to_dict(orient="records"),
        "unstandardized_results": unstandardized_results.to_dict(orient="records"),
        "standardized_coefs": standardized_coefs.tolist(),
        "gammas": gammas.tolist(),
        "feature_cols": feature_cols,
        "preprocessing": {
            "station_categories": station_categories,
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "use_weekend": False,
        },
        "best_standardized": best_std.to_dict(),
        "best_unstandardized": best_unstd.to_dict(),
        "comparison": {
            "test-rmse_difference": float(best_std["test-rmse"] - best_unstd["test-rmse"]),
            "test-r2_difference": float(best_std["test-r2"] - best_unstd["test-r2"]),
        },
        "top_features": [
            {"feature": feature_cols[j], "coefficient": float(best_coefs[j])}
            for j in top_idx
        ],
    }
    # Serialize before opening the destination, so invalid numbers don't erase it.
    serialized = json.dumps(saved, indent=2, allow_nan=False)
    Path(path).write_text(serialized + "\n", encoding="utf-8")


def load_results(path=SAVED_RESULTS_PATH):
    """Restore the arrays and tables needed to reproduce the printed report/plot."""
    with Path(path).open(encoding="utf-8") as file:
        saved = json.load(file)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported saved results format")
    return (
        saved["linear_results"],
        pd.DataFrame(saved["standardized_results"]),
        pd.DataFrame(saved["unstandardized_results"]),
        np.asarray(saved["standardized_coefs"], dtype=float),
        np.asarray(saved["gammas"], dtype=float),
        saved["feature_cols"],
    )


def main(load=False, saved_path=SAVED_RESULTS_PATH):
    """Example driver showing how to read the CitiBike CSVs, engineer
    features, and evaluate a model. This is a starting point for all the
    written analysis part of the problem. You can extend it in main or
    use a different file to call the functions.

    Assumes the data sits in the current working directory. With load=True,
    reproduce the saved report and plot without reading data or fitting models.
    """
    started = perf_counter()

    def progress(message):
        print(f"[elapsed {perf_counter() - started:.1f}s] {message}", flush=True)

    if load:
        progress(f"Loading saved results from {Path(saved_path).resolve()}")
        saved = load_results(saved_path)
        progress("Saved results loaded; printing report and plotting")
        report_results(*saved, progress=progress)
        progress("Finished")
        return

    progress("Reading training CSV")
    train_df = pd.read_csv("citibike_2022_train_processed.csv", low_memory=False)
    progress(f"Training CSV loaded: {len(train_df):,} rows; reading test CSV")
    test_df = pd.read_csv("citibike_2023_test_processed.csv", low_memory=False)
    progress(f"Test CSV loaded: {len(test_df):,} rows")

    # graph_correlations(train_df) 

    # handle unseen stations
    progress("Learning station categories from training data")
    station_categories = fit_station_categories(train_df)
    progress("Encoding training features")
    train_df = preprocess_features(train_df, station_categories=station_categories)
    progress("Encoding test features")
    test_df = preprocess_features(test_df, station_categories=station_categories)
    progress("Aligning feature columns")
    # Keep the training station vocabulary and column order for prediction.
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    feature_cols = [c for c in train_df.columns if c not in ("duration_min", "ride_id")]
    progress(f"Converting {len(feature_cols):,} features to NumPy arrays")
    train_y = train_df["duration_min"].to_numpy()
    test_y = test_df["duration_min"].to_numpy()
    train_x = train_df[feature_cols].to_numpy(dtype=float)
    test_x = test_df[feature_cols].to_numpy(dtype=float)

    progress("Standardizing training and test arrays")
    train_x_std, test_x_std = standardize_data(train_x, test_x)
    progress("Standardization complete")

    # TODO: Do all of the written parts of the problem below.

    # Q1d. run eval_linear here
    progress("Q1d: fitting and evaluating linear regression (updates resume when it finishes)")
    linear_started = perf_counter()
    results = eval_linear(train_x_std, train_y, test_x_std, test_y)
    progress(f"Q1d complete in {perf_counter() - linear_started:.1f}s")

    # 20 log-spaced gamma values from 10^-4 to 10^10
    gammas = np.logspace(-4, 10, 20)

    # Standardized data.
    # Save coefficients here because Q1h needs these exact same fitted models.
    progress("Starting standardized ridge sweep: 20 gammas")
    standardized_results, standardized_coefs = ridge_sweep(
        train_x_std,
        train_y,
        test_x_std,
        test_y,
        gammas,
        save_coefs=True,
        progress=progress,
        label="Standardized ridge",
    )

    # Unstandardized data.
    # Q1h does not need these coefficients, so don't save them.
    progress("Starting unstandardized ridge sweep: 20 gammas")
    unstandardized_results = ridge_sweep(
        train_x,
        train_y,
        test_x,
        test_y,
        gammas,
        progress=progress,
        label="Unstandardized ridge",
    )
    progress("Computing preprocessing statistics for saving")
    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0)
    train_std = np.where(train_std == 0, 1.0, train_std)
    progress(f"Writing results to {Path(saved_path).resolve()}")
    save_results(
        saved_path, results, standardized_results, unstandardized_results,
        standardized_coefs, gammas, feature_cols, station_categories,
        train_mean, train_std,
    )
    progress("Results saved; printing report and plotting")
    report_results(
        results, standardized_results, unstandardized_results,
        standardized_coefs, gammas, feature_cols, progress=progress,
    )
    progress("Finished")


def report_results(results, standardized_results, unstandardized_results,
                   standardized_coefs, gammas, feature_cols, progress=None):
    """Print all metrics and draw Q1h using only saved results, with no fitting."""
    print("printing linear regression results. Q1d")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("\nQ1e, f: Ridge regression")
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

    # Q1h: ridge coefficient path
    print("\nQ1h: Ridge coefficient path")

    # Find the gamma selected in Q1f
    best_idx = standardized_results["test-rmse"].idxmin()
    best_gamma = standardized_results.loc[best_idx, "gamma"]

    # Coefficients at the selected gamma
    best_coefs = standardized_coefs[best_idx]

    # Find the 3 largest coefficients in absolute value at the selected gamma
    top_idx = np.argsort(np.abs(best_coefs))[-3:][::-1]

    print("Top 3 features at selected gamma:")
    for j in top_idx:
        print(
            f"{feature_cols[j]}: "
            f"coefficient = {best_coefs[j]:.6f}"
        )

    # Plot every coefficient path
    if progress:
        progress(f"Drawing coefficient paths for {len(feature_cols):,} features")
    plt.figure(figsize=(12, 8))

    for j in range(len(feature_cols)):
        plt.plot(
            gammas,
            standardized_coefs[:, j],
            linewidth=2 if j in top_idx else 0.7,
            alpha=1.0 if j in top_idx else 0.4,
            label=feature_cols[j]
        )

    # Mark the gamma selected in Q1f
    plt.axvline(
        best_gamma,
        linestyle="--",
        linewidth=2,
        label=f"Selected gamma = {best_gamma:.2e}"
    )

    plt.xscale("log")
    plt.xlabel("Gamma")
    plt.ylabel("Ridge coefficient")
    plt.title("Ridge Coefficient Paths (Standardized Data)")
    plt.tight_layout()
    # Include every feature once. Keep the potentially large legend outside
    # the axes; bbox_inches="tight" includes the entire legend in the output.
    legend_columns = min(8, max(1, int(np.ceil((len(feature_cols) + 1) / 35))))
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        ncol=legend_columns,
        fontsize=6,
        title="All features (top 3 paths emphasized)",
    )

    if progress:
        progress("Saving q1h_ridge_coefficient_path.png")
    plt.savefig(
        "q1h_ridge_coefficient_path.png",
        # Thousands of legend entries produce a large image even at 100 dpi.
        dpi=100 if len(feature_cols) > 200 else 300,
        bbox_inches="tight"
    )

    if progress:
        progress("Plot saved; opening plot (if a window appears, close it to finish)")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and save Q1 results, or replay saved results.")
    parser.add_argument(
        "--load", action="store_true",
        help="Read saved_coef.txt beside q1.py without reading CSVs or fitting models.",
    )
    args = parser.parse_args()
    main(load=args.load)
