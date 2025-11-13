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

def perform_statistical_test(final_rewards):
    """Performs statistical tests and returns formatted results."""
    from scipy import stats
    
    adam_rewards = final_rewards["Adam"]
    radam_rewards = final_rewards["RAdam"]
    
    # Perform normality test
    _, adam_norm_p = stats.shapiro(adam_rewards)
    _, radam_norm_p = stats.shapiro(radam_rewards)
    
    # Choose test based on normality results
    if adam_norm_p > 0.05 and radam_norm_p > 0.05:
        # Parametric t-test
        t_stat, p_value = stats.ttest_ind(adam_rewards, radam_rewards)
        test_type = "Independent t-test"
    else:
        # Non-parametric Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(adam_rewards, radam_rewards)
        test_type = "Mann-Whitney U test"
    
    # Calculate effect size
    def cohen_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    effect_size = cohen_d(adam_rewards, radam_rewards)
    
    return {
        "p_value": p_value,
        "effect_size": effect_size,
        "test_type": test_type,
        "adam_mean": np.mean(adam_rewards),
        "radam_mean": np.mean(radam_rewards),
        "adam_ci": stats.t.interval(0.95, len(adam_rewards)-1, loc=np.mean(adam_rewards), scale=stats.sem(adam_rewards)),
        "radam_ci": stats.t.interval(0.95, len(radam_rewards)-1, loc=np.mean(radam_rewards), scale=stats.sem(radam_rewards))
    }

def create_stat_test_plot(final_rewards, output_path_base="stat_test_plot"):
    """Creates statistical test plot with consistent styling."""
    if not final_rewards or len(final_rewards["Adam"]) < 3 or len(final_rewards["RAdam"]) < 3:
        print("Not enough data for statistical test")
        return

    test_results = perform_statistical_test(final_rewards)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

def create_ftest_plot(final_rewards, output_path_base="f_test_plot"):
    """Creates F-test plot for equality of variances with consistent styling."""
    from scipy.stats import f
    
    # Extract rewards
    adam_rewards = final_rewards["Adam"]
    radam_rewards = final_rewards["RAdam"]
    
    # Check sample size requirements
    if len(adam_rewards) < 2 or len(radam_rewards) < 2:
        print("Insufficient data for F-test")
        return

    # Calculate variances with Bessel's correction
    var_adam = np.var(adam_rewards, ddof=1)
    var_radam = np.var(radam_rewards, ddof=1)
    
    # Ensure F-statistic >= 1
    if var_adam >= var_radam:
        f_stat = var_adam / var_radam
        df1 = len(adam_rewards) - 1
        df2 = len(radam_rewards) - 1
    else:
        f_stat = var_radam / var_adam
        df1 = len(radam_rewards) - 1
        df2 = len(adam_rewards) - 1

    # Calculate two-tailed p-value
    p_value = (1 - f.cdf(f_stat, df1, df2))

    # Setup plot with consistent styling
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 3))
    
    # Create F-distribution curve
    x = np.linspace(0, f.ppf(0.999, df1, df2), 1000)
    y = f.pdf(x, df1, df2)
    plt.plot(x, y, color='blue', linewidth=1.4, 
             label=f'F({df1}, {df2}) Distribution')
    
    # Add critical value marker
    plt.axvline(f_stat, color='red', linestyle='--', linewidth=1.2)
    
    # Shade critical region
    critical_x = np.linspace(f_stat, x[-1], 300)
    plt.fill_between(critical_x, f.pdf(critical_x, df1, df2),
                     color='blue', alpha=0.4)
    
    # Format p-value in scientific notation
    p_value_str = f'{p_value:.1e}'  # Use scientific notation with one decimal place

    # Add annotation with matching style
    plt.text(f_stat * 1.05, np.max(y) * 0.85,
            f'F = {f_stat:.2f}\np = {p_value_str}',
            fontsize=12, color='red', va='center')
    
    # Style elements matching KDE plot
    plt.xlabel("F Value", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    plt.title("F-test of Equality of Variances", fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, frameon=True, loc='upper right')
    
    # Final styling touches
    sns.despine()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    # Save with same naming convention
    plt.savefig(f"{output_path_base}_Adam_RAdam_2.svg", dpi=300)
    plt.show()

if __name__ == '__main__':
    csvs_dir = "./Combined_CSVs"
    final_rewards = extract_all_final_rewards(csvs_dir)

    if final_rewards:
        #create_kde_plots(final_rewards, kernels=['gaussian'])
        create_ftest_plot(final_rewards)  # Add this line to generate F-test plot
    else:
        print("No data was found.")