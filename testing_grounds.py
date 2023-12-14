import pandas as pd
import numpy as np
import os
import time
import plotly.graph_objects as go
import logging
from SALib.sample import sobol
from scipy.stats import spearmanr, rankdata, norm
import matplotlib.pyplot as plt
import cProfile

import dash_functions_and_callbacks as dfc
import uncertainties


"""
# define path to farm files folder and create list with dataframes from the farm files
farm_files_path = "farm files/"
farm_paths_list = os.listdir(farm_files_path)
farm_data_df_list = []

for filename in farm_paths_list:
    # Create the full file path by concatenating the directory and the filename
    full_file_path = os.path.join(farm_files_path, filename)
    # Use the full file path to read the data into a dataframe
    dataframe = dfc.read_file_to_dataframe(full_file_path)
    if isinstance(dataframe, pd.DataFrame):
        farm_data_df_list.append(dataframe)
    else:
        print(dataframe)  # This will print the error message if the file could not be processed

animals_on_farms = []

for farm_df in farm_data_df_list:
    # Iterate over each row in the dataframe
    for index, row in farm_df.iterrows():
        # Check if the number of animals for this type is greater than 0
        if row['num-animals'] > 0:
            # Add the animal type to the list, if not already present
            animal = row['name']
            if animal not in animals_on_farms:
                animals_on_farms.append(animal)



animal_config_samples_path = "monte_carlo_simulation/saved_animal_samples.xlsx"
environmental_config_samples_path = "monte_carlo_simulation/saved_env_samples.xlsx"


env_config_samples = pd.read_excel(environmental_config_samples_path)
animal_config_samples = pd.read_excel(animal_config_samples_path)


# Initialize a dictionary to hold the aggregated data
aggregated_data = {}

# Iterate over each row
for index, row in animal_config_samples.iterrows():
    animal_type = row['animal_type']
    stable_type = row['stable_type']

    # Check if the animal type is one of those we're interested in
    if animal_type in animals_on_farms:
        unique_key = f"{animal_type}_{stable_type}"

        # Initialize a sub-dictionary for this unique combination if it doesn't exist
        if unique_key not in aggregated_data:
            aggregated_data[unique_key] = {}

        # Aggregate data for each column
        for column in animal_config_samples.columns:
            if column not in ['animal_type', 'stable_type']:
                new_name = f"{unique_key}_{column}"
                # Aggregate the values
                if new_name not in aggregated_data[unique_key]:
                    aggregated_data[unique_key][new_name] = []
                aggregated_data[unique_key][new_name].append(row[column])
    else:
        # Skip the rows where animal_type is not in animals_on_farms
        continue  # Or handle differently if you need to include these rows with NaN values or similar

# Ensure all lists in aggregated_data are of the same length before creating a DataFrame
max_length = max(len(lst) for subdict in aggregated_data.values() for lst in subdict.values())
for subdict in aggregated_data.values():
    for key, lst in subdict.items():
        if len(lst) < max_length:
            lst.extend([np.nan] * (max_length - len(lst)))  # Fill with NaNs to match the max length

# Convert the aggregated data into a DataFrame
new_columns = {k: v for d in aggregated_data.values() for k, v in d.items()}
new_animal_samples = pd.DataFrame(new_columns)

new_animal_samples.to_excel('Debug/new_animal_samples.xlsx', index=False)

# Combine the DataFrames
combined_samples = pd.concat([env_config_samples, new_animal_samples], axis=1)

combined_samples.to_excel("Debug/combined_samples.xlsx", index=False)


sim_results_dict = uncertainties.load_from_hdf5()

result_key = 'co2_eq_tot'
treatment_results = {}

for treatment, df_list in sim_results_dict.items():
    treatment_results[treatment] = [dfc.find_value_in_results_df(df, result_key) for df in df_list]

print(treatment_results)



# Convert your input variables and output results to ranks
ranked_inputs = combined_samples.apply(rankdata)
ranked_outputs = {treatment: rankdata(results) for treatment, results in treatment_results.items()}

# Initialize a dictionary to store PRCC results
prcc_results = {treatment: {} for treatment in treatment_results}

# Function to calculate confidence interval for Spearman correlation
def spearman_ci(rho, n, alpha=0.05):
    # Fisher's Z transformation for rho
    z = np.arctanh(rho)
    # Standard error
    se = 1 / np.sqrt(n - 3)
    # Z critical value
    z_crit = norm.ppf(1 - alpha/2)
    # Margin of error
    moe = z_crit * se
    # Confidence interval
    z_ci_lower, z_ci_upper = z - moe, z + moe
    # Transform back to correlation scale
    rho_ci_lower, rho_ci_upper = np.tanh(z_ci_lower), np.tanh(z_ci_upper)
    return rho_ci_lower, rho_ci_upper

# Compute Spearman rank correlation for each treatment
for treatment, results in ranked_outputs.items():
    n = len(results)  # Number of data points
    for column in ranked_inputs.columns:
        if np.std(ranked_inputs[column]) == 0 or np.std(results) == 0:
            continue  # One of the arrays is constant; correlation is not defined
        else:
            correlation, p_value = spearmanr(ranked_inputs[column], results)
            ci_lower, ci_upper = spearman_ci(correlation, n)  # Calculate CI
            prcc_results[treatment][column] = {
                'correlation': correlation,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }

# prcc_results now contains the PRCC, p-values, and confidence intervals for each input variable

print(f'prcc results: \n{prcc_results}')

# Determine the number of subplots needed
num_treatments = len(prcc_results)
fig_height_per_subplot = 5  # Adjust height as needed
fig, axes = plt.subplots(num_treatments, 1, figsize=(10, fig_height_per_subplot * num_treatments))

for treatment, variables in prcc_results.items():
    # Filter variables based on p-value
    filtered_variables = {var: values for var, values in variables.items() if values['p_value'] <= 0.1}

    # Continue only if there are any variables left after filtering
    if not filtered_variables:
        print(f"No variables with p-value <= 0.2 for treatment {treatment}. Skipping plot.")
        continue

    correlations = [info['correlation'] for info in filtered_variables.values()]
    lower_bounds = [info['ci_lower'] for info in filtered_variables.values()]
    upper_bounds = [info['ci_upper'] for info in filtered_variables.values()]
    variable_names = list(filtered_variables.keys())

    # Calculate the error amounts for each point
    # The error amounts should be positive values
    lower_errors = [corr - lower for corr, lower in zip(correlations, lower_bounds)]
    upper_errors = [upper - corr for upper, corr in zip(upper_bounds, correlations)]

    # Combine the lower and upper errors into a list of tuples
    error_bars = [lower_errors, upper_errors]  # Two rows: one for lower and one for upper errors

    # Create figure for each treatment
    plt.figure(figsize=(10, len(variable_names) * 0.5))  # Adjust figure size as needed
    plt.errorbar(correlations, range(1, len(variable_names) + 1), xerr=error_bars, fmt='o')
    plt.yticks(range(1, len(variable_names) + 1), variable_names)
    plt.xlabel('PRCC')
    plt.title(f'PRCC Values for {treatment}')
    plt.grid(True)

    # Improve layout
    plt.tight_layout()

    # Save the figure with a unique name for each treatment
    plt.savefig(f'Debug/prcc_plot_{treatment}.png')  # Change the path if necessary
    plt.show()

"""






"""
Code for improving efficiency


def main():
    # define path to animal and environmental config
    default_animal_config_path = "default_configs/default_animal_config.xlsx"
    default_environmental_config_path = "default_configs/default_environmental_config.xlsx"

    # define path to farm files folder and create list with dataframes from the farm files
    farm_files_path = "farm files/"
    farm_paths_list = os.listdir(farm_files_path)
    farm_data_df_list = []

    for filename in farm_paths_list:
        # Create the full file path by concatenating the directory and the filename
        full_file_path = os.path.join(farm_files_path, filename)
        # Use the full file path to read the data into a dataframe
        dataframe = dfc.read_file_to_dataframe(full_file_path)
        if isinstance(dataframe, pd.DataFrame):
            farm_data_df_list.append(dataframe)
        else:
            print(dataframe)  # This will print the error message if the file could not be processed

    # Read the configuration files
    animal_config = dfc.read_file_to_dataframe(default_animal_config_path)
    env_config = dfc.read_file_to_dataframe(default_environmental_config_path)

    no_treatment_df, ad_only_df, ad_biogas_df, steam_ad_df, steam_ad_biogas_df = uncertainties.lca_calculation(
        env_config, animal_config)


if __name__ == "__main__":
    cProfile.run('main()', 'Debug/profile_output')
"""


