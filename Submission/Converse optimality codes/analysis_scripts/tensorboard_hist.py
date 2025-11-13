import os
import pandas as pd
import matplotlib.pyplot as plt
def extract_final_reward_from_csv(csv_path, threshold=-18000): # Add threshold parameter
    """Extracts the final reward from a CSV file, filtering values below the threshold."""
    try:
        df = pd.read_csv(csv_path)
        if not df.empty:
            if 'Value' in df.columns:
                final_reward = df['Value'].iloc[-1]
            elif 'ep_rew_mean' in df.columns:
                final_reward = df['ep_rew_mean'].iloc[-1]
            else:
                print(f"Error: 'Value' or 'ep_rew_mean' column not found in {csv_path}")
                return None
            
            if final_reward >= threshold: # Apply filter directly after reading
                return final_reward
            else:
                return None # Return None if below the threshold

        else:
            print(f"Warning: CSV file {csv_path} is empty.")
            return None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None


def extract_all_final_rewards(csvs_dir):
    """Extracts final reward values from all relevant CSV files, filtering out values less than -80000."""
    all_final_rewards = {
        "Adam": [],
        "RAdam": [],
    }
    
    for filename in os.listdir(csvs_dir):
        if "Adam" in filename and filename.endswith(".csv"):
            filepath = os.path.join(csvs_dir, filename)
            final_reward = extract_final_reward_from_csv(filepath)
            if final_reward is not None:
                if "__Adam" in filename and final_reward >= -180000: # Filter condition for Adam
                    all_final_rewards["Adam"].append(final_reward)
                elif "_RAdam" in filename and final_reward >= -180000: # Filter condition for RAdam
                    all_final_rewards["RAdam"].append(final_reward)
    
    return all_final_rewards


def create_histograms(final_rewards, output_path_base="histogram"):
    """Generates histograms from final reward values."""
    if not final_rewards:
        print("No data to plot.")
        return

    num_plots = len(final_rewards)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), sharey=True) # Adjust figsize as needed
    fig.suptitle("Histograms of Final Reward Values for Different Optimizers", fontsize=16)


    for i, (optimizer, rewards) in enumerate(final_rewards.items()):
        ax = axes[i] if num_plots > 1 else axes # Handle single plot case
        ax.hist(rewards, bins=10, alpha=0.7, edgecolor='black') # Adjust bins as needed
        ax.set_xlabel("Final Reward Value")
        ax.set_ylabel("Frequency")
        ax.set_title(optimizer)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(f"{output_path_base}_Adam_RAdam2_All.png")
    plt.show()




if __name__ == '__main__':
    csvs_dir = "./Combined_CSVs" # Replace with the path to your CSV folder
    final_rewards = extract_all_final_rewards(csvs_dir)
    print(final_rewards["RAdam"])

    if final_rewards:
        create_histograms(final_rewards) # Create histograms
    else:
        print("No data was found.")

