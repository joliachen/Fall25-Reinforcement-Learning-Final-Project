#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 1. Load CSV
    csv_path = "wandb_export.csv"  
    df = pd.read_csv(csv_path)

    # 2. Extract step column
    if "Step" not in df.columns:
        raise ValueError("Could not find 'Step' column in CSV.")
    steps = df["Step"].values

    # 3. Find columns corresponding to reward_mean for each run
    #    columns like: "2025.11.29_09.59.54 - reward_mean"
    reward_cols = [c for c in df.columns if c.endswith(" - reward_mean")]

    if len(reward_cols) == 0:
        raise ValueError("No columns ending with ' - reward_mean' found.")

    # 4. Compute mean and std across runs for each step
    reward_values = df[reward_cols].to_numpy()  # shape: (num_steps, num_runs)
    mean_rewards = reward_values.mean(axis=1)
    std_rewards = reward_values.std(axis=1, ddof=1)  # sample std across runs

    # 5. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_rewards, label="Mean reward over runs")
    plt.fill_between(
        steps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        label="±1 std dev",
    )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Mean ± Std of reward_mean over runs")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reward_mean_std_over_runs.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
