import requests
import os

# This file is used to automatically download CSV files from a tensorboard run.

base_url = "http://localhost:6006/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag=rollout%2Fep_rew_mean&run=test__Adam_opti_run_40__38_1&format=csv"


def download_tensorboard_data(run_name, output_dir="./tensorboard_40more_csvs"):
    """Downloads TensorBoard data for a specific run."""

    url = f"{base_url}&run={run_name}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        filename = os.path.join(output_dir, f"{run_name}.csv") # Create filename
        os.makedirs(output_dir, exist_ok=True) #Ensure output dir exists

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): #Stream the download to avoid memory issues
                f.write(chunk)
        print(f"Downloaded data for {run_name} to {filename}")


    except requests.exceptions.RequestException as e:
        print(f"Error downloading data for {run_name}: {e}")
    except Exception as e: #Catch any other potential errors
        print(f"An unexpected error occurred for {run_name}: {e}")

# --- Main Execution: Define your runs and download ---

runs = [                                      #List of run names
    "test_RAdam_opti_run_" + str(i) + "_1" for i in range(40)
]
runs.extend(["test__Adam_opti_run_" + str(i) + "_1" for i in range(40)])

for run in runs:
    download_tensorboard_data(run)
