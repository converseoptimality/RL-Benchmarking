# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# def extract_final_reward_from_csv(csv_path):
#     """Extracts the final reward from a CSV file."""
#     try:
#         df = pd.read_csv(csv_path)
#         if not df.empty:
#             if 'Value' in df.columns:
#                 return df['Value'].iloc[-1]
#             elif 'ep_rew_mean' in df.columns:
#                 return df['ep_rew_mean'].iloc[-1]
#             else:
#                 print(f"Error: 'Value' or 'ep_rew_mean' column not found in {csv_path}")
#                 return None
#         else:
#             print(f"Warning: CSV file {csv_path} is empty.")
#             return None
#     except Exception as e:
#         print(f"Error reading CSV file {csv_path}: {e}")
#         return None


# def extract_all_final_rewards(csvs_dir):
#     """Extracts final reward values from all relevant CSV files."""
#     all_final_rewards = {
#         "Adam": [],
#         "RAdam": [],
#         #"0.95": []
#     }
    
#     for filename in os.listdir(csvs_dir):
#         if "Adam" in filename and filename.endswith(".csv"):
#             filepath = os.path.join(csvs_dir, filename)
#             final_reward = extract_final_reward_from_csv(filepath)
#             if final_reward is not None:
#                 if "__Adam" in filename:
#                     all_final_rewards["Adam"].append(final_reward)
#                 elif "_RAdam" in filename:
#                     all_final_rewards["RAdam"].append(final_reward)
#                 #elif "0.95" in filename:
#                 #    all_final_rewards["0.95"].append(final_reward)
    
#     return all_final_rewards


# def create_box_plot(final_rewards, output_path="box_plot.png"):
#     """Generates a box plot from final reward values demonstrating locality, spread and skewness."""
#     if not final_rewards:
#         print("No data to plot.")
#         return

#     plt.figure(figsize=(10, 6))  # Set the figure size
#     # Create a box plot for the three thresholds
#     box_plot = plt.boxplot(final_rewards.values(), showmeans=False, patch_artist=True, showfliers=False)

#     # Set the tick labels
#     plt.xticks(range(1, len(final_rewards) + 1), final_rewards.keys())
    
#     plt.ylabel("Final Reward Value")
#     plt.title("Box Plot of Final Reward Values (0.0, 0.5)")
#     plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for the y-axis
#     plt.tight_layout()  # Automatically adjust plot parameters 

#     # Save and display the plot
#     plt.savefig(output_path)
#     plt.show()


# if __name__ == '__main__':
#     csvs_dir = "./Combined_CSVs"  # Replace with the path to your CSV folder
#     final_rewards = extract_all_final_rewards(csvs_dir)
#     print(final_rewards["RAdam"])

#     if final_rewards:
#         create_box_plot(final_rewards, output_path="box_plot_Adam_RAdam_All.png")  # Output the plot as a svg file
#     else:
#         print("No data was found.")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_final_reward_from_csv(csv_path):
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


def create_box_plot(final_rewards, output_path="box_plot.png"):
    """Generates a box plot from final reward values with similar styling to KDE plots."""
    if not final_rewards:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))

    data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in final_rewards.items()])) # Convert dict to pandas DataFrame

    #colors = {"Adam": "#e41a1c", "RAdam": "#377eb8"} # Consistent color palette

    # Create the boxplot using seaborn
    ax = sns.boxplot(data=data, showmeans=False, showfliers=False, linewidth=1.0, width=0.3)

    # Customize box appearance
    for box in ax.artists:
        box.set_edgecolor('black')
        box.set_linewidth(1)

    # Customize whisker and cap appearance
    for i, line in enumerate(ax.lines):
        line.set_color('black')
        line.set_linewidth(1)

    # Customize median line appearance
    for median in ax.lines[4::6]:
        median.set_color('#ff7f0e')  # Bright orange color
        median.set_linewidth(2.5)    # Slightly thicker line


    plt.ylabel("Final Reward", fontsize=14)
    plt.title("Final Reward Box Plot", fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12) # Increase x-tick font size
    plt.yticks(fontsize=12) # Increase y-tick font size

    sns.despine()  # Remove top and right spines

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    csvs_dir = "./Combined_CSVs"
    final_rewards = extract_all_final_rewards(csvs_dir)
    print(final_rewards["RAdam"])

    if final_rewards:
        create_box_plot(final_rewards, output_path="box_plot_Adam_RAdam_Styled.svg")
    else:
        print("No data was found.")
