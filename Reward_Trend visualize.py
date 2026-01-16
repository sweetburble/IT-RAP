import re
import matplotlib.pyplot as plt
import pandas as pd


def extract_rewards_from_file(file_path):
    rewards = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"Episode\s+\d+:\s+([0-9.]+)", line)
            if match:
                reward = float(match.group(1))
                rewards.append(reward)
    return rewards


file_paths = [

    ("C:\\Users\\Bandi\\Desktop\\Fork\\IT-RAP\\test_result_images\\CelebA + StarGAN 1\\reward_moving_avg.txt", "CelebA 100 images | AttGAN", "tab:green"),

    ("C:\\Users\\Bandi\\Desktop\\Fork\\IT-RAP\\test_result_images\\MAAD + StarGAN 1\\reward_moving_avg.txt", "MAAD 100 images | AttGAN", "tab:blue")
]

window_size = 50
plt.figure(figsize=(14, 6))


for path, label, color in file_paths:
    rewards = extract_rewards_from_file(path)
    series = pd.Series(rewards)


    mean = series.mean()
    std = series.std()
    normalized = (series - mean) / std


    smoothed = normalized.rolling(window=window_size, min_periods=1, center=True).mean()


    plt.plot(smoothed.index, smoothed.values, label=f"{label}", color=color, linewidth=2)


plt.title("Reward Trend Over Episodes", fontsize=20)
plt.xlabel("Episode", fontsize=20)
plt.ylabel("Normalized Reward", fontsize=20)


plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc='upper left', fontsize=18)
plt.tight_layout()
plt.savefig("reward_trend.png")
plt.show()
