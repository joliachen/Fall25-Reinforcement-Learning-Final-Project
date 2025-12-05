#!/usr/bin/env python3
"""Visualize DDPO reward statistics for successful and failed hyperparameters.

This script reads two Weights & Biases CSV exports:

    - ``visualize_training/wandb_export.csv`` (successful hyperparameters)
    - ``visualize_training/wandb_export_failed.csv`` (failed hyperparameters)

For each file, it:
    1. Extracts the training steps from the ``"Step"`` column.
    2. Finds all columns whose names end with ``" - reward_mean"`` and treats
       each of them as one independent run.
    3. Computes the mean and standard deviation of ``reward_mean`` across runs
       at each step.

It then generates a single PNG figure with two subplots placed horizontally:
    - Left: Successful hyperparameters (mean ± std reward curves)
    - Right: Failed hyperparameters (mean ± std reward curves)

The resulting figure is saved to ``doc/reward_mean_std_compare.png``.
"""

from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_mean_std(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a W&B CSV export and compute mean/std of reward over runs.

    The CSV file is expected to contain:
        * A ``"Step"`` column indicating training step or global step.
        * One or more columns whose names end with ``" - reward_mean"``.
          Each such column is interpreted as the ``reward_mean`` time series
          for one independent run (e.g., different random seeds).

    This function aggregates across all ``reward_mean`` columns by computing
    the mean and sample standard deviation at each step.

    Args:
        csv_path: Path to the CSV file exported from Weights & Biases.

    Returns:
        steps: A 1D NumPy array of shape ``(num_steps,)`` with the step values.
        mean_rewards: A 1D NumPy array of shape ``(num_steps,)`` with the mean
            reward across runs for each step.
        std_rewards: A 1D NumPy array of shape ``(num_steps,)`` with the sample
            standard deviation of the reward across runs for each step.

    Raises:
        ValueError: If the CSV does not contain a ``"Step"`` column or if no
            columns ending with ``" - reward_mean"`` can be found.
    """
    # Load the CSV into a pandas DataFrame.
    df = pd.read_csv(csv_path)

    # Ensure the required "Step" column is present.
    if "Step" not in df.columns:
        raise ValueError(f"CSV {csv_path} does not contain 'Step' column.")

    # Extract the step values as a NumPy array.
    steps = df["Step"].values

    # Identify all columns that represent reward_mean from different runs.
    reward_cols = [c for c in df.columns if c.endswith(" - reward_mean")]
    if len(reward_cols) == 0:
        raise ValueError(f"No columns ending with ' - reward_mean' found in {csv_path}")

    # Shape: (num_steps, num_runs)
    reward_values = df[reward_cols].to_numpy()

    # Compute the mean reward across runs for each step.
    mean_rewards = reward_values.mean(axis=1)

    # Compute the sample standard deviation (ddof=1) across runs for each step.
    std_rewards = reward_values.std(axis=1, ddof=1)

    return steps, mean_rewards, std_rewards


def plot_mean_std(
    ax: plt.Axes,
    steps: np.ndarray,
    mean_rewards: np.ndarray,
    std_rewards: np.ndarray,
    title: str,
) -> None:
    """Plot mean ± standard deviation of reward on a given Matplotlib axis.

    This helper function draws a line plot of the mean reward over steps and
    shades the area corresponding to ±1 standard deviation.

    Args:
        ax: A Matplotlib ``Axes`` object on which to draw the plot.
        steps: A 1D NumPy array containing the step values.
        mean_rewards: A 1D NumPy array containing the mean reward per step.
        std_rewards: A 1D NumPy array containing the standard deviation of
            reward per step (same shape as ``mean_rewards``).
        title: Title string for the subplot.
    """
    # Plot the mean reward trajectory.
    ax.plot(steps, mean_rewards, label="Mean reward")

    # Fill the area representing ±1 standard deviation.
    ax.fill_between(
        steps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        label="±1 std",
    )

    # Set axis labels and title.
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(title)

    # Add a light grid and legend for readability.
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    """Entry point for generating the comparison plot.

    This function:
        1. Loads statistics from the CSV with successful hyperparameters.
        2. Loads statistics from the CSV with failed hyperparameters.
        3. Creates a figure with two horizontally arranged subplots.
        4. Plots mean ± std reward curves for both cases.
        5. Saves the resulting figure as a PNG file.

    The expected input files are:
        - ``visualize_training/wandb_export.csv``
        - ``visualize_training/wandb_export_failed.csv``

    The output file is:
        - ``doc/reward_mean_std_compare.png``
    """
    # Paths to the W&B export CSV files.
    success_csv = "visualize_training/wandb_export.csv"
    failed_csv = "visualize_training/wandb_export_failed.csv"

    # Compute mean/std curves for successful and failed hyperparameters.
    steps_s, mean_s, std_s = compute_mean_std(success_csv)
    steps_f, mean_f, std_f = compute_mean_std(failed_csv)

    # Create a 1×2 grid of subplots (horizontally aligned).
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: successful hyperparameters.
    plot_mean_std(
        axes[0],
        steps_s,
        mean_s,
        std_s,
        title="Successful Hyperparameters",
    )

    # Right subplot: failed hyperparameters.
    plot_mean_std(
        axes[1],
        steps_f,
        mean_f,
        std_f,
        title="Failed Hyperparameters",
    )

    # Adjust layout to avoid overlapping labels and titles.
    plt.tight_layout()

    # Save the figure to disk.
    plt.savefig("doc/reward_mean_std_compare.png", dpi=200)

    # Optionally display the figure in an interactive session.
    plt.show()


if __name__ == "__main__":
    main()
