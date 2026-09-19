"""Run Q3's six elastic net experiments; use --load to replot saved losses."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from elastic import ElasticNet
from q1 import SAVED_RESULTS_PATH, preprocess_features


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "q3_training_objectives.json"


def plot_results(results):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    for ax, run in zip(axes.flat, results["runs"]):
        history = run["history"]
        ax.plot([int(epoch) for epoch in history], list(history.values()))
        ax.set_title(f"{run['dataset']} (n={run['n']:,}), eta={run['eta']:.0e}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training objective")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Elastic net: alpha={results['alpha']}, gamma={results['gamma']:.8g}, "
        f"batch={results['batch']}"
    )
    fig.tight_layout()
    output = ROOT / "q3_training_objectives.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}", flush=True)


def main(load=False):
    if load:
        plot_results(json.loads(RESULTS.read_text()))
        return

    saved = json.loads(SAVED_RESULTS_PATH.read_text())
    best = min(saved["standardized_results"], key=lambda row: row["test-rmse"])
    gamma = float(best["gamma"])
    alpha, batch, epochs, fraction, seed = 0.5, 1024, 30, 0.05, 534
    # Keep the saved gamma unchanged. Its large L2 curvature requires small steps.
    learning_rates = np.logspace(-7, -5, 3)
    results = {
        "gamma": gamma, "alpha": alpha, "batch": batch,
        "max_epochs": epochs, "subsample_fraction": fraction, "seed": seed,
        "learning_rates": learning_rates.tolist(), "runs": [],
    }
    metadata = saved["preprocessing"]
    features = saved["feature_cols"]
    csv_path = ROOT / "citibike_2022_train_processed.csv"
    y = pd.read_csv(csv_path, usecols=["duration_min"])["duration_min"].to_numpy(dtype=float)
    mean = np.asarray(metadata["train_mean"])
    std = np.asarray(metadata["train_std"])
    started = perf_counter()

    # A disk-backed array avoids allocating the entire dense design in RAM.
    with TemporaryDirectory(prefix="q3_standardized_") as directory, threadpool_limits(limits=1):
        x = np.memmap(Path(directory) / "x.dat", dtype=float, mode="w+",
                      shape=(len(y), len(features)))
        offset = 0
        for chunk in pd.read_csv(csv_path, chunksize=10000, low_memory=False):
            frame = preprocess_features(
                chunk, use_weekend=metadata["use_weekend"],
                station_categories=metadata["station_categories"],
            )
            values = frame.reindex(columns=features, fill_value=0).to_numpy(dtype=float)
            x[offset:offset + len(chunk)] = (values - mean) / std
            offset += len(chunk)
        x.flush()
        print(f"Standardized {x.shape} in {perf_counter() - started:.1f}s", flush=True)

        subset = np.random.default_rng(seed).choice(len(y), int(len(y) * fraction), replace=False)
        sub_x, sub_y = x[subset], y[subset]
        for label, train_x, train_y in [("5% subsample", sub_x, sub_y), ("Full training set", x, y)]:
            for eta in learning_rates:
                print(f"Training {label}, eta={eta:.0e}, gamma={gamma:.8g}", flush=True)
                # Same initial parameters and shuffle sequence for each rate.
                np.random.seed(seed)
                model = ElasticNet(gamma, alpha, float(eta), batch, epochs)
                run_started = perf_counter()
                history = model.train(train_x, train_y)
                results["runs"].append({
                    "dataset": label, "n": len(train_y), "eta": float(eta),
                    "history": history, "intercept": model.beta_intercept,
                    "nonzero_coefficients": int(np.count_nonzero(model.beta)),
                })
                RESULTS.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
                print(f"Finished {len(history)} epochs in {perf_counter() - run_started:.1f}s; "
                      f"final objective={history[len(history)]:.6f}", flush=True)
        del train_x, x
    plot_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true", help="Replot the saved six runs without training")
    main(load=parser.parse_args().load)
