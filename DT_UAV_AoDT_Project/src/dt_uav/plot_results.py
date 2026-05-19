import argparse
import os

import pandas as pd


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return plt


def _save_bar(df, column, ylabel, path, budget=None):
    plt = _plt()
    summary = df.groupby("policy")[column].mean().sort_values()
    ax = summary.plot(kind="bar", color="#3b6ea8", edgecolor="black", linewidth=0.7)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if budget is not None:
        ax.axhline(budget, color="#b33a3a", linestyle="--", linewidth=1.2, label="budget")
        ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _save_line(csv_path, y_cols, ylabel, path):
    plt = _plt()
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    ax = df.plot(x="update", y=y_cols)
    ax.set_xlabel("PPO update")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_results(out_dir, budget=None):
    eval_path = os.path.join(out_dir, "evaluation.csv")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")

    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    df = pd.read_csv(eval_path)
    _save_bar(df, "avg_aodt", "Average AoDT (slots)", os.path.join(figs_dir, "avg_aodt.png"))
    _save_bar(df, "p95_aodt", "95th percentile AoDT (slots)", os.path.join(figs_dir, "p95_aodt.png"))
    _save_bar(df, "max_aodt", "Maximum AoDT (slots)", os.path.join(figs_dir, "max_aodt.png"))
    _save_bar(
        df,
        "mean_backhaul_energy",
        "Mean backhaul energy (J/slot/UAV)",
        os.path.join(figs_dir, "backhaul_energy.png"),
        budget=budget,
    )
    if "served_ratio" in df.columns:
        _save_bar(df, "served_ratio", "Successful update ratio", os.path.join(figs_dir, "served_ratio.png"))

    _save_line(
        os.path.join(out_dir, "worker_train_log.csv"),
        ["avg_aodt", "served_per_slot", "invalid_per_slot"],
        "Worker metric",
        os.path.join(figs_dir, "worker_training.png"),
    )
    _save_line(
        os.path.join(out_dir, "manager_train_log.csv"),
        ["avg_aodt", "mean_backhaul_energy", "max_Z"],
        "Manager metric",
        os.path.join(figs_dir, "manager_training.png"),
    )

    print(f"Saved figures to {figs_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--budget", type=float, default=None)
    args = parser.parse_args()
    plot_results(args.out, budget=args.budget)


if __name__ == "__main__":
    main()
