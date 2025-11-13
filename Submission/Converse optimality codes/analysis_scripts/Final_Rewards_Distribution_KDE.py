import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
def extract_final_reward_from_csv(csv_path):
    """Extracts the final reward from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if not df.empty:
            if 'Value' in df.columns:
                return df['Value'].iloc[-1]
            elif 'ep_rew_mean' in df.columns:
                return df['ep_rew_mean'].iloc[-1]
            else:
                print(f"Error: 'Value' or 'ep_rew_mean' column not found in {csv_path}")
                return None
        else:
            print(f"Warning: CSV file {csv_path} is empty.")
            return None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None


def extract_all_final_rewards(csvs_dir):
    """Extracts final reward values from all relevant CSV files."""
    all_final_rewards = {
        "Adam": [],
        "RAdam": [],
    }

    for filename in os.listdir(csvs_dir):
        if "Adam" in filename and filename.endswith(".csv"):
            filepath = os.path.join(csvs_dir, filename)
            final_reward = extract_final_reward_from_csv(filepath)
            if final_reward is not None:
                if "__Adam" in filename:
                    all_final_rewards["Adam"].append(final_reward)
                elif "_RAdam" in filename:
                    all_final_rewards["RAdam"].append(final_reward)

    return all_final_rewards
def create_kde_plots(final_rewards, kernels=['epanechnikov'], output_path_base="kde_plot"):
    """Generates visually appealing and scientific KDE plots."""

    if not final_rewards:
        print("No data to plot.")
        return

    sns.set_style("whitegrid") # Set a visually appealing style
    plt.figure(figsize=(6, 3)) # Adjust figure size

    colors = {"Adam": "blue", "RAdam": "green"} # Use distinct, colorblind-friendly colors
    line_styles = {"Adam": "-", "RAdam": "--"} # define line styles for each optimizer

    for optimizer, rewards in final_rewards.items():
        rewards = np.array(rewards).reshape(-1, 1)
        for kernel in kernels:
            kde = KernelDensity(kernel=kernel, bandwidth=80).fit(rewards)
            x_vals = np.linspace(min(rewards) - 500, max(rewards) + 500, 2000).reshape(-1, 1) # Slightly extended range
            y_vals = np.exp(kde.score_samples(x_vals))


            plt.plot(x_vals, y_vals, label=f"{optimizer} ({kernel})", linewidth=1.4, color=colors[optimizer], linestyle=line_styles[optimizer])
            plt.fill_between(x_vals.flatten(), 0, y_vals, alpha=0.4, color=colors[optimizer]) # Semi-transparent fill


    plt.xlabel("Final Reward", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    plt.title("Final Rewards Distribution (KDE)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, title="Optimizers", title_fontsize=12) # Improved legend

    # Remove top and right spines for a cleaner look
    sns.despine()


    plt.tight_layout()
    plt.savefig(f"{output_path_base}_Adam_RAdam_edited.svg", dpi=300) # Higher resolution
    plt.show()



if __name__ == '__main__':
    csvs_dir = "./Combined_CSVs"
    final_rewards = extract_all_final_rewards(csvs_dir)

    if final_rewards:
        create_kde_plots(final_rewards, kernels=['gaussian'])  # Explicitly use gaussian kernel
    else:
        print("No data was found.")

