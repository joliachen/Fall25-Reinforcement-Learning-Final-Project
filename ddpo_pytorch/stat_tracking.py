import numpy as np
from collections import deque


class PerPromptStatTracker:
    """Track per-prompt reward statistics and compute normalized advantages.

    This helper maintains a rolling buffer of past rewards for each prompt
    string and uses them to normalize current rewards into advantages:

        advantage = (reward - mean) / std

    where `mean` and `std` are computed either from the global batch or from
    the per-prompt buffer, depending on how many samples have been seen for
    that prompt.
    """
    def __init__(self, buffer_size, min_count):
        """Initialize the per-prompt statistics tracker.

        Args:
            buffer_size: Maximum number of past rewards to keep per prompt.
                When this limit is reached, older rewards are discarded in
                FIFO order.
            min_count: Minimum number of stored rewards required before using
                per-prompt statistics for normalization. If fewer than
                `min_count` rewards are stored for a prompt, the tracker falls
                back to using the batch-level mean and std.
        """
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        """Update statistics and compute normalized advantages for a batch.

        Args:
            prompts: Iterable of prompt identifiers (typically strings), one per reward.
            rewards: Iterable of scalar rewards, same length as `prompts`.

        Returns:
            `np.ndarray` of advantages with the same shape as `rewards`.
            For each unique prompt, rewards are normalized using either:
            * the global batch mean/std (if fewer than `min_count` past rewards
              have been stored for that prompt), or
            * the per-prompt rolling mean/std, once enough samples are
              available.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        """Get current per-prompt summary statistics.

        Returns:
            dict: A mapping from prompt to a dictionary with keys:
                * `"mean"`: Mean reward over the stored buffer.
                * `"std"`: Standard deviation of rewards over the buffer.
                * `"count"`: Number of rewards currently stored.
        """
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }
