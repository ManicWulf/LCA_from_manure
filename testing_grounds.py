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



# Combine the DataFrames
combined_samples = pd.concat([env_config_samples, new_animal_samples], axis=1)




sim_results_dict = uncertainties.load_from_hdf5()

result_key = 'co2_eq_tot'
treatment_results = {}

for treatment, df_list in sim_results_dict.items():
    treatment_results[treatment] = [dfc.find_value_in_results_df(df, result_key) for df in df_list]

print(treatment_results)



# Convert your input variables and output results to ranks
ranked_inputs = combined_samples.apply(rankdata)
ranked_outputs = {treatment: rankdata(results) for treatment, results in treatment_results.items()}

# Function to calculate confidence interval for Spearman correlation
def bootstrap_spearman_ci(ranked_inputs, ranked_outputs, num_bootstraps=1000, confidence_level=0.95):
    bootstrap_correlations = []

    for _ in range(num_bootstraps):
        bootstrap_results = {}
        # Resampling the data
        for treatment, results in ranked_outputs.items():
            resampled_results = np.random.choice(results, size=len(results), replace=True)
            bootstrap_results[treatment] = resampled_results

        raw_correlations = []
        # Recalculating correlations for the resampled data
        for treatment, results in bootstrap_results.items():
            for column in ranked_inputs.columns:
                if np.std(ranked_inputs[column]) == 0 or np.std(results) == 0:
                    continue
                correlation, _ = spearmanr(ranked_inputs[column], results)
                raw_correlations.append(correlation)

        # Normalizing the correlations
        sum_of_squares = sum(c**2 for c in raw_correlations)
        normalized_correlations = [c**2 / sum_of_squares for c in raw_correlations]
        bootstrap_correlations.extend(normalized_correlations)

    # Calculating the confidence intervals
    lower_bound = np.percentile(bootstrap_correlations, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_correlations, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound

def bootstrap_spearman_ci_single_pair(input_column, output_results, num_bootstraps=1000, confidence_level=0.95):
    bootstrap_correlations = []

    for _ in range(num_bootstraps):
        # Resample the data with replacement
        resampled_results = np.random.choice(output_results, size=len(output_results), replace=True)
        correlation, _ = spearmanr(input_column, resampled_results)
        bootstrap_correlations.append(correlation)

    # Calculating the confidence intervals
    lower_bound = np.percentile(bootstrap_correlations, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_correlations, (1 + confidence_level) / 2 * 100)

    # Normalize the confidence interval bounds
    sum_of_squares = sum(c**2 for c in bootstrap_correlations)
    lower_bound_normalized = np.sqrt(lower_bound ** 2 / sum_of_squares) if not np.isnan(lower_bound) else np.nan
    upper_bound_normalized = np.sqrt(upper_bound ** 2 / sum_of_squares) if not np.isnan(upper_bound) else np.nan

    return lower_bound_normalized, upper_bound_normalized


def bootstrap_spearman_ci_single_pair_non_parametric(input_column, output_results, num_bootstraps=1000, confidence_level=0.95):
    bootstrap_correlations = []

    for _ in range(num_bootstraps):
        # Resample the data with replacement
        resampled_results = np.random.choice(output_results, size=len(output_results), replace=True)
        correlation, _ = spearmanr(input_column, resampled_results)
        bootstrap_correlations.append(correlation)

    # Non-parametric confidence intervals using the percentile method
    lower_bound = np.percentile(bootstrap_correlations, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_correlations, (1 + confidence_level) / 2 * 100)

    # Plotting the distribution of bootstrap correlations
    plt.hist(bootstrap_correlations, bins=30, alpha=0.7, color='blue')
    plt.title("Distribution of Bootstrap Correlations")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.show()

    # Print descriptive statistics
    print("Mean Correlation:", np.mean(bootstrap_correlations))
    print("Median Correlation:", np.median(bootstrap_correlations))
    print("Standard Deviation:", np.std(bootstrap_correlations))

    return lower_bound, upper_bound

    return lower_bound, upper_bound


# Initialize a dictionary to store PRCC results
prcc_results = {treatment: {} for treatment in ranked_outputs}

# Loop for computing Spearman rank correlation for each treatment and input column
for treatment, results in ranked_outputs.items():
    raw_correlations = []
    for column in ranked_inputs.columns:
        if np.std(ranked_inputs[column]) == 0 or np.std(results) == 0:
            continue  # Skip if one of the arrays is constant

        # Compute Spearman rank correlation
        correlation, p_value = spearmanr(ranked_inputs[column], results)
        raw_correlations.append(correlation)

    # Filter out NaN values from raw_correlations
    filtered_correlations = [c for c in raw_correlations if not np.isnan(c)]

    # Calculate the sum of the squares of the non-NaN correlations for normalization
    sum_of_squares = sum(c ** 2 for c in filtered_correlations)

    # Check if sum_of_squares is zero (which can happen if all correlations were NaN)
    if sum_of_squares == 0:
        print(f"All correlations are NaN for treatment: {treatment}")
        continue

    # Normalize each correlation and calculate CI for each column
    for i, correlation in enumerate(raw_correlations):
        column = ranked_inputs.columns[i]
        normalized_correlation = np.sqrt(correlation ** 2 / sum_of_squares) if not np.isnan(correlation) else np.nan
        ci_lower, ci_upper = bootstrap_spearman_ci_single_pair_non_parametric(ranked_inputs[column], results)

        # Store the normalized correlation and CI in prcc_results
        prcc_results[treatment][column] = {
            'correlation': normalized_correlation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }




# Check the final prcc_results for NaN values
for treatment, variables in prcc_results.items():
    for column, values in variables.items():
        if np.isnan(values['correlation']):
            print(f"NaN stored in prcc_results for treatment: {treatment}, column: {column}")

# Directory for saving plots
save_directory = "plots"  # Replace with your actual directory

# Determine the number of subplots needed
num_treatments = len(prcc_results)
fig_height_per_subplot = 5  # Adjust height as needed

for treatment, variables in prcc_results.items():
    # Filter variables based on normalized correlation value
    filtered_variables = {var: values for var, values in variables.items() if abs(values['correlation']) >= 0.1}

    if not filtered_variables:
        print(f"No significant correlations for treatment {treatment}. Skipping plot.")
        continue

    correlations = [info['correlation'] for info in filtered_variables.values()]
    lower_bounds = [info['ci_lower'] for info in filtered_variables.values()]
    upper_bounds = [info['ci_upper'] for info in filtered_variables.values()]
    variable_names = list(filtered_variables.keys())

    lower_errors = [abs(corr - lower) for corr, lower in zip(correlations, lower_bounds)]
    upper_errors = [max(0, upper - corr) for upper, corr in zip(upper_bounds, correlations)]

    print(f"Treatment: {treatment}")
    print("Correlations:", correlations)
    print("Lower Errors:", lower_errors)
    print("Upper Errors:", upper_errors)

    error_bars = [lower_errors, upper_errors]

    plt.figure(figsize=(10, max(2, len(variable_names) * 0.5)))
    plt.errorbar(correlations, range(len(variable_names)), xerr=error_bars, fmt='o')
    plt.yticks(range(len(variable_names)), variable_names)
    plt.xlabel('Normalized PRCC')
    plt.title(f'Normalized PRCC Values for {treatment}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_directory}/prcc_plot_{treatment}.png')
    plt.show()






