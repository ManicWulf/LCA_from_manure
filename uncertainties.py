import pandas as pd
import numpy as np
import os
import time
import plotly.graph_objects as go
import logging
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, as_completed



import farm_calc as fc
import dash_functions_and_callbacks as dfc


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',  # This will log to a file named app.log. Remove this to log to console.
                    filemode='w')  # This means the log file will be overwritten each time the app is started. Use 'a' to append.


# define path to animal and environmental config
default_animal_config_path = "default_configs/default_animal_config.xlsx"
default_environmental_config_path = "default_configs/default_environmental_config.xlsx"
combined_config_path = 'default_configs/updated_combined_config.xlsx'

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

"""



Change the value for sample_size to adjust the number of simulation runs




"""


# Define the sample_size variable
sample_size = 15000  # You can adjust this value as needed


def write_dict_to_csv(simulation_results_dict, file_path):
    with open(file_path, 'w', newline='') as file:
        for treatment, df_list in simulation_results_dict.items():
            # Write treatment type as a header
            file.write(f"\n{treatment}: {treatment}\n")

            # Write each DataFrame in the list to the CSV file
            for df in df_list:
                # If the DataFrame is empty, skip writing it
                if df.empty:
                    continue
                df.to_csv(file, index=False)
                file.write("\n")  # Add a newline to separate DataFrames


def save_to_hdf5(simulation_results_dict, filename="simulation_results.h5"):
    filepath = f'monte_carlo_simulation/{filename}'
    with pd.HDFStore(filepath, mode='w') as store:
        for treatment, df_list in simulation_results_dict.items():
            for idx, df in enumerate(df_list):
                # Prepend 'sim_' to the key to make it a valid identifier
                key = f'{treatment}_{idx}'
                store.put(key, df, format='table')



def load_from_hdf5(filepath='monte_carlo_simulation/simulation_results.h5'):
    results = {}
    with pd.HDFStore(filepath, mode='r') as store:
        # Custom sorting function to handle the new key format
        def sorting_key_func(x):
            parts = x.lstrip('/').rsplit('_', 1)  # Splitting by the last underscore
            if len(parts) > 1 and parts[1].isdigit():
                return (parts[0], int(parts[1]))
            else:
                return (parts[0], x)

        sorted_keys = sorted(store.keys(), key=sorting_key_func)
        for key in sorted_keys:
            treatment, _ = key.lstrip('/').rsplit('_', 1)  # Splitting by the last underscore
            if treatment not in results:
                results[treatment] = []
            df = store[key]
            results[treatment].append(df)
    return results




# function to combine the calculations from farm_calc.py for the monte carlo simulation
def lca_calculation(env_config, animal_config, list_farms=None):
    start_time = time.time()
    # 1. Perform farm calculations for all farms

    logging.info('Start of farm calculations \n \n \n')
    if list_farms is None:
        list_farms = farm_data_df_list
    farm_animal_calc = fc.calc_all_farms_new(list_farms, animal_config, env_config)

    farm_animal_calc.to_excel('Debug/farm_animal_calc.xlsx', index=False)

    # 2. Perform the no-treatment calculations using farm_animal_calc

    logging.info('Start of no treatment calculations \n \n \n')
    no_treatment_df = fc.post_storage_and_field_emissions(farm_animal_calc, env_config, False)
    no_treatment_df = fc.calc_env_impacts(no_treatment_df, env_config)

    # set ad and chp construction emissions to 0 for the no treatment scenario
    dfc.store_value_in_results_df(no_treatment_df, "co2_ad_construction", 'value', 0)
    dfc.store_value_in_results_df(no_treatment_df, "co2_chp_construction", 'value', 0)

    no_treatment_df.to_excel('Debug/no_treatment.xlsx', index=False)

    # 3. Perform AD only calculations

    logging.info('Start of AD only calculations \n \n \n')
    ad_only_df = fc.calc_anaerobic_digestion(farm_animal_calc, env_config)
    ad_only_df = fc.post_storage_and_field_emissions(ad_only_df, env_config, True)
    ad_only_df = fc.calc_chp_output(ad_only_df, env_config)
    ad_only_df = fc.calc_env_impacts(ad_only_df, env_config)

    ad_only_df.to_excel('Debug/ad_only.xlsx', index=False)

    # 4. Perform biogas upgrading calculations for AD+Biogas based on ad_only_df

    logging.info('Start of AD biogas calculations \n \n \n')
    ad_biogas_df = fc.calc_biogas_upgrading(ad_only_df, env_config)
    ad_biogas_df = fc.calc_env_impacts(ad_biogas_df, env_config)

    ad_biogas_df.to_excel('Debug/ad_biogas.xlsx', index=False)

    # 5. Calculate steam pretreatment with AD based on farm_animal_calc

    logging.info('Start of Steam AD calculations \n \n \n')
    steam_ad_df = fc.steam_pre_treatment(farm_animal_calc, env_config)
    #steam_ad_df.to_excel('Debug/steam_ad_pre_treatment.xlsx', index=False)
    steam_ad_df = fc.calc_anaerobic_digestion(steam_ad_df, env_config)
    #steam_ad_df.to_excel('Debug/steam_ad_anaerobic_digestion.xlsx', index=False)
    steam_ad_df = fc.post_storage_and_field_emissions(steam_ad_df, env_config, True)
    #steam_ad_df.to_excel('Debug/steam_ad_post_field.xlsx', index=False)
    steam_ad_df = fc.calc_chp_output(steam_ad_df, env_config)
    #steam_ad_df.to_excel('Debug/steam_ad_chp.xlsx', index=False)
    steam_ad_df = fc.calc_env_impacts(steam_ad_df, env_config)

    steam_ad_df.to_excel('Debug/steam_ad_env_impact.xlsx', index=False)

    # 6. Calculate biogas upgrading for steam_ad_df

    logging.info('Start of Steam AD biogas calculations \n \n \n')
    steam_ad_biogas_df = fc.calc_biogas_upgrading(steam_ad_df, env_config)
    steam_ad_biogas_df = fc.calc_env_impacts(steam_ad_biogas_df, env_config)

    steam_ad_biogas_df.to_excel('Debug/steam_ad_biogas.xlsx', index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time} seconds")
    return no_treatment_df, ad_only_df, ad_biogas_df, steam_ad_df, steam_ad_biogas_df


def create_results_list_monte_carlo(simulation_results_dict, no_treatment, ad_only, ad_biogas, steam_ad, steam_ad_biogas):

    # Check if the dictionary is empty
    if not simulation_results_dict:
        # Initialize the dictionary with the function parameters as keys and the arguments as the first list items
        simulation_results_dict = {
            'no_treatment': [no_treatment],
            'ad_only': [ad_only],
            'ad_biogas': [ad_biogas],
            'steam_ad': [steam_ad],
            'steam_ad_biogas': [steam_ad_biogas]
        }
    else:
        # Add the input values to the lists in the dictionary
        simulation_results_dict['no_treatment'].append(no_treatment)
        simulation_results_dict['ad_only'].append(ad_only)
        simulation_results_dict['ad_biogas'].append(ad_biogas)
        simulation_results_dict['steam_ad'].append(steam_ad)
        simulation_results_dict['steam_ad_biogas'].append(steam_ad_biogas)

    return simulation_results_dict



# Function to generate lognormal distribution based on mean and standard deviation
def generate_lognormal(mean, std, size = 1):
    """
    Generate a lognormal distribution given the mean, standard deviation and number of samples.
    mean and std could be a single value or an array.
    """
    if std == 0 or np.isnan(std):
        return np.full(size, mean)
    else:
        # Calculate the parameters for the lognormal distribution
        sigma = np.sqrt(np.log(1 + (std/mean)**2))
        mu = np.log(mean) - sigma**2 / 2
        return np.random.lognormal(mu, sigma, size)


def generate_lognormal_median_k(median, k, size = 1):
    """
    Generate a lognormal distribution given the median, dispersion factor k, and number of samples.
    """
    if k <= 1:
        return np.full(size, median)
    else:
        sigma = np.log(k)
        mu = np.log(median)
        return np.random.lognormal(mu, sigma, size)


def generate_lognormal_min_max_median(min_value, max_value, median, size=1):
    """
    Generate a lognormal distribution given the minimum, maximum, and median values.

    :param min_value: The minimum value of the distribution
    :param max_value: The maximum value of the distribution
    :param median: The median value of the distribution
    :param size: The number of samples to generate
    :return: An array of samples from the lognormal distribution
    """
    if min_value >= max_value or min_value <= 0:
        raise ValueError("Invalid min and max values. Ensure min < max and min > 0.")

    # Calculate mu from the median
    mu = np.log(median)

    # Assume min and max are the 5th and 95th percentiles
    z_5 = norm.ppf(0.05)  # z-score for 5th percentile
    z_95 = norm.ppf(0.95)  # z-score for 95th percentile

    # Solving for sigma using min and max values
    sigma = (np.log(max_value) - np.log(min_value)) / (z_95 - z_5)

    # Generate the lognormal distribution
    return np.random.lognormal(mu, sigma, size)


def generate_lognormal_standard_error_sample_size_median(se, median, n, size=1):
    """
    Generate a lognormal distribtion given the standard error and median
    :param se: standard error of the distribution
    :param median: median of the distribution
    :param n: sample size of the data source
    :param size: the number of samples to generate
    :return: An array of samples from the lognormal distribution
    """
    # Calculate mu from the median
    mu = np.log(median)

    # Calculate sigma from se
    sigma = calculate_sigma_from_se(se, n)

    # Generate the lognormal distribution
    return np.random.lognormal(mu, sigma, size)


def generate_lognormal_mu_sigma(mu, sigma, size=1):
    """
    Generate a lognormal distribution given the mu and sigma values
    :param mu: mu of the lognormal distribution
    :param sigma: sigma of the lognormal distribution
    :param size: number of samples to generate
    :return: An arreay of samples from the lognormal distribution
    """

    # Generate the lognormal distribution
    return np.random.lognormal(mu, sigma, size)


def generate_triangle(min_value, max_value, median, size = 1):
    """
    Generate a triangle distribution given the minimum, maximum, and median values.
    """
    return np.random.triangular(min_value, median, max_value, size)


def calculate_sigma_from_se(se, n):
    """
    calculate sigma of a lognormal distribution based on standard error and sample size
    :param se: standard error
    :param n: sample size
    :return: sigma
    """
    return se * np.sqrt(n)


def extract_uncertainty_parameters(farm_df):
    """
    Extracts the uncertainty parameters (pre-storage, post-storage, distance) from a new farm file using the find_value_farm function.

    Parameters:
    farm_df (DataFrame): The dataframe of a new farm file.

    Returns:
    dict: A dictionary containing the min, max, and expected values for pre-storage, post-storage, and distance.
    """
    uncertainty_params = {}

    # Extract rows for each parameter
    for param in ['pre_storage', 'post_storage', 'distance']:
        min_val = dfc.find_value_farm(f'{param}_min', 'additional-data', farm_df)
        expected_val = dfc.find_value_farm(f'{param}_expected', 'additional-data', farm_df)
        max_val = dfc.find_value_farm(f'{param}_max', 'additional-data', farm_df)

        # Convert string values to float
        min_val = int(min_val) if min_val is not None else None
        expected_val = int(expected_val) if expected_val is not None else None
        max_val = int(max_val) if max_val is not None else None

        # Store in dictionary
        uncertainty_params[param] = {
            'min': min_val,
            'expected': expected_val,
            'max': max_val
        }

    return uncertainty_params


def generate_random_values_for_farm(uncertainty_params, num_samples=sample_size):
    """
    Generates random values for each uncertainty parameter (pre-storage, post-storage, distance)
    using a triangular distribution.

    Parameters:
    uncertainty_params (dict): Dictionary containing the min, max, and expected values for each parameter.
    num_samples (int): Number of random samples to generate for each parameter.

    Returns:
    dict: A dictionary containing arrays of generated samples for each parameter.
    """
    random_values = {}
    for param, values in uncertainty_params.items():
        min_val = values['min']
        expected_val = values['expected']
        max_val = values['max']

        # Generate random samples using triangular distribution
        samples = generate_triangle(min_val, max_val, expected_val, num_samples)
        random_values[param] = samples

    return random_values



def substitute_env_config_values(samples_df, default_config, iteration):
    """
    Substitute values in the default environmental configuration with values from a sample row.

    :param samples_df: The generated samples dataframe.
    :param default_config: The default environmental configuration dataframe.
    :param iteration: Index of the row in samples_df to use for substitution.
    :return: A new dataframe with substituted values.
    """
    # Create a copy of the default configuration to prevent in-place modification
    changed_config = default_config.copy()

    # Iterate over each parameter in the default config
    for index, row in changed_config.iterrows():
        param_name = row['name']
        if param_name in samples_df.columns:
            # Use a specific sample for each iteration
            sample_value = samples_df.at[iteration, param_name]
            changed_config.at[index, 'value'] = sample_value
        else:
            # If parameter not found in samples, keep the original value
            continue

    return changed_config


def substitute_animal_config_values(samples_df, default_config, iteration):
    # Make a copy of the default_config to avoid modifying the original DataFrame
    changed_config = default_config.copy()

    # Convert 'stable_type' in changed_config to integer if it's not already
    changed_config['stable_type'] = changed_config['stable_type'].astype(int)

    # Filter the samples for the current iteration
    samples_for_iteration = samples_df[samples_df['iteration'] == iteration]

    # Convert 'stable_type' in samples_for_iteration to integer if it's not already
    samples_for_iteration['stable_type'] = pd.to_numeric(samples_for_iteration['stable_type'], errors='coerce').fillna(0).astype(int)

    # Debugging: Print types for columns of interest
    #print(f"Changed config types: {changed_config[['animal_type', 'stable_type']].dtypes}")
    #print(f"Samples for iteration types: {samples_for_iteration[['animal_type', 'stable_type']].dtypes}")

    # Check if we found any samples for the current iteration
    if samples_for_iteration.empty:
        raise ValueError(f"No samples found for iteration {iteration}.")

    # Get the list of columns that are common to both DataFrames and should be updated
    common_columns = set(samples_df.columns) & set(changed_config.columns)
    columns_to_update = common_columns - {'animal_type', 'stable_type', 'iteration'}

    # Debugging: Print the columns that will be updated
    #print(f"Columns to update: {columns_to_update}")

    # Iterate over each unique combination of animal_type and stable_type in changed_config
    for animal, stable in changed_config[['animal_type', 'stable_type']].drop_duplicates().itertuples(index=False):
        #print(f"Looking for animal: {animal}, stable: {stable}")
        # Find the corresponding row in samples_for_iteration
        sample_row = samples_for_iteration[
            (samples_for_iteration['animal_type'] == animal) &
            (samples_for_iteration['stable_type'] == stable)
        ]

        if sample_row.empty:
            print(f"No matching sample found for animal: {animal}, stable: {stable}")
        else:
            #print(f"Found matching sample for animal: {animal}, stable: {stable}")
            sample_row = sample_row.iloc[0]  # Assuming there's only one match, take the first row
            for col in columns_to_update:
                # Update the changed_config with the sample value
                changed_config.loc[
                    (changed_config['animal_type'] == animal) &
                    (changed_config['stable_type'] == stable), col
                ] = sample_row[col]

    return changed_config




def substitute_farm_files_values(farm_data_dict, farm_samples_dict,n):
    # Update each farm file with the generated random samples
    updated_farm_files = []
    for farm_file_name, farm_df in farm_data_dict.items():
        farm_samples = farm_samples_dict[farm_file_name]
        updated_farm_df = update_farm_with_samples(farm_df, farm_samples, n)
        updated_farm_files.append(updated_farm_df)


    return updated_farm_files


def single_simulation_run(n, df_animal_samples, df_env_samples, env_config, animal_config, farm_data_dict, farm_samples_dict):
    animal_config_df = substitute_animal_config_values(df_animal_samples, animal_config, n)
    env_config_df = substitute_env_config_values(df_env_samples, env_config, n)
    updated_farm_files = substitute_farm_files_values(farm_data_dict, farm_samples_dict,n)


    # Run the LCA calculation and return its result along with the simulation index
    return n, lca_calculation(env_config_df, animal_config_df, updated_farm_files)


def create_config_for_simulation(df_animal_samples, df_env_samples, env_config, animal_config, num_simulations,
                                 list_farms=None, farm_samples_dict=None):
    simulation_results_dict = {}

    try:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(single_simulation_run, n, df_animal_samples, df_env_samples, env_config, animal_config,
                                farm_data_dict, farm_samples_dict)
                for n in range(num_simulations)]

            for future in as_completed(futures):
                try:
                    n, result = future.result()
                    simulation_results_dict = create_results_list_monte_carlo(simulation_results_dict, *result)
                except Exception as e:
                    print(f"Exception in future: {e}")

            # Shut down the executor explicitly
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Exception in create_config_for_simulation: {e}")

    save_to_hdf5(simulation_results_dict)
    return simulation_results_dict


def create_config_for_simulation_old(df_animal_samples, df_env_samples, env_config, num_simulations, list_farms=None, farm_samples_dict=None):
    simulation_results_dict = {}
    for n in range(num_simulations):
        # Select one sample per unique combination of animal type and manure type
        animal_config_df = df_animal_samples.drop_duplicates(subset=['animal_type', 'stable_type'])

        # If similar logic is needed for environmental samples, do it here
        env_config_df = substitute_env_config_values(df_env_samples, env_config, n)

        # Update each farm file with the generated random samples
        updated_farm_files = substitute_farm_files_values(farm_data_dict, n)


        # Use config_df in your model here

        simulation_results_dict = create_results_list_monte_carlo(simulation_results_dict, *lca_calculation(env_config_df, animal_config_df, updated_farm_files))

        # Optionally, save config_df to check its structure
        #animal_config_df.to_excel(f"Monte carlo simulation/animal_config_test{n}.xlsx", index=False)
        #env_config_df.to_excel(f"Monte carlo simulation/env_config_test{n}.xlsx", index=False)

    save_to_hdf5(simulation_results_dict)

    return simulation_results_dict


def update_farm_with_samples(farm_df, farm_samples, iteration):
    """
    Updates the farm file dataframe with generated random samples for a specific iteration.
    Creates new rows for 'pre_storage', 'post_storage', and 'distance' if they don't exist.

    Parameters:
    farm_df (DataFrame): The dataframe of a farm file.
    farm_samples (dict): Dictionary containing arrays of generated samples for each parameter.
    iteration (int): The iteration index to select the specific sample.

    Returns:
    DataFrame: Updated farm file dataframe.
    """
    updated_farm_df = farm_df.copy()

    # Update the dataframe with the sample values for the given iteration
    for param in ['pre_storage', 'post_storage', 'distance']:
        sample_value = farm_samples[param][iteration]
        sample_value = int(sample_value)

        # Check if the row exists, and update or create it accordingly
        if param in updated_farm_df['name'].values:
            param_row_index = updated_farm_df[updated_farm_df['name'] == param].index
            updated_farm_df.at[param_row_index[0], 'additional-data'] = sample_value
        else:
            # Create a new row for the parameter
            new_row = pd.DataFrame({'name': [param], 'additional-data': [sample_value]})
            updated_farm_df = pd.concat([updated_farm_df, new_row], ignore_index=True)

    return updated_farm_df


def split_combined_config(combined_config):
    """
    Splits the combined configuration into environmental and animal configurations.

    Parameters:
    combined_config (DataFrame): The combined configuration DataFrame.

    Returns:
    tuple: A tuple containing two DataFrames (environmental configuration, animal configuration).
    """
    # Criteria for identifying animal configuration rows
    animal_config_pattern = r'_0_|_1_|_2_'

    # Environmental Configuration: Exclude rows that match the animal configuration pattern
    env_config_columns = ['name', 'median', 'upper', 'lower', 'sigma', 'mu', 'Distribution function']
    env_config = combined_config[~combined_config['name'].str.contains(animal_config_pattern, na=False)]
    env_config = env_config[env_config_columns]

    # Animal Configuration: Include only rows that match the animal configuration pattern
    animal_config = combined_config[combined_config['name'].str.contains(animal_config_pattern, na=False)]

    # Exporting environmental configuration to Excel for debugging
    env_config.to_excel("Debug/env_config_extracted.xlsx", index=False)

    return env_config, animal_config


def transform_animal_config(animal_config):
    """
    Transforms the animal configuration to include mu and sigma for each parameter,
    split by animal type and stable type, with mu and sigma columns adjacent to each other.

    Parameters:
    animal_config (DataFrame): The animal configuration DataFrame.

    Returns:
    DataFrame: The transformed animal configuration DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    animal_config_copy = animal_config.copy()

    # Define a function to extract the animal type, stable type, and parameter
    def extract_info(name):
        parts = name.split('_')
        # Find the index of the stable type (always a single digit)
        stable_type_idx = next(i for i, part in enumerate(parts) if part in ['0', '1', '2'])
        animal_type = '_'.join(parts[:stable_type_idx])
        stable_type = parts[stable_type_idx]
        parameter = '_'.join(parts[stable_type_idx + 1:])
        return animal_type, stable_type, parameter

    # Apply the function to each row in the DataFrame
    extracted_info = animal_config_copy['name'].apply(
        lambda x: pd.Series(extract_info(x), index=['animal_type', 'stable_type', 'parameter']))
    animal_config_copy = pd.concat([animal_config_copy, extracted_info], axis=1)

    # Pivot the DataFrame to create mu and sigma columns for each parameter
    animal_config_pivoted = animal_config_copy.pivot_table(
        index=['animal_type', 'stable_type'],
        columns='parameter',
        values=['mu', 'sigma'],
        aggfunc='first'
    )

    # Flatten the MultiIndex in columns
    animal_config_pivoted.columns = ['_'.join(col[::-1]) for col in animal_config_pivoted.columns]

    # Reorder columns so that each mu_parameter is next to its corresponding sigma_parameter
    sorted_columns = sorted(animal_config_pivoted.columns, key=lambda x: (x.split('_')[1], x.split('_')[0]))
    animal_config_pivoted = animal_config_pivoted[sorted_columns]

    animal_config_pivoted.reset_index(inplace=True)

    animal_config_pivoted.to_excel("Debug/animal_config_pivoted.xlsx", index=False)

    return animal_config_pivoted







def plot_histograms_plotly(sim_results_dict, result_key='co2_eq_tot'):
    # Create a Plotly figure
    fig = go.Figure()

    # Prepare a dictionary to hold frequency data
    frequency_data = {}

    # Loop through each treatment in the simulation_results_dict to calculate frequencies
    for treatment, df_list in sim_results_dict.items():
        values = [dfc.find_value_in_results_df(df, result_key) for df in df_list]
        frequency_data[treatment] = pd.Series(values).value_counts().sum()

    # Sort treatments by frequency (lowest to highest)
    sorted_treatments = sorted(frequency_data, key=frequency_data.get, reverse=True)

    # Define a list of distinct colors for the histograms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    # Loop through each treatment in sorted order to plot histograms
    for i, treatment in enumerate(sorted_treatments):
        values = [dfc.find_value_in_results_df(df, result_key) for df in sim_results_dict[treatment]]
        fig.add_trace(go.Histogram(
            x=values,
            name=treatment,
            opacity=0.6,  # Adjust the opacity
            marker_color=colors[i % len(colors)]  # Assign color from the list
        ))

    # Update layout for the figure
    fig.update_layout(
        title=f'Histograms of CO2 Equivalents Total for Different Treatments with {sample_size} Simulations',
        xaxis_title='CO2 Equivalents Total',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay histograms
        bargap=0.2,  # Gap between bars of adjacent location coordinates
        bargroupgap=0.1  # Gap between bars of the same location coordinate
    )

    # Show the plot
    fig.show()




def plot_boxplots_plotly(sim_results_dict, result_key='co2_eq_tot'):
    # Create a Plotly figure
    fig = go.Figure()

    # Define a list of distinct colors for the boxplots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Loop through each treatment in the simulation_results_dict to plot boxplots
    for i, (treatment, df_list) in enumerate(sim_results_dict.items()):
        values = [dfc.find_value_in_results_df(df, result_key) for df in df_list]
        fig.add_trace(go.Box(
            y=values,
            name=treatment,
            marker_color=colors[i % len(colors)]  # Assign color from the list
        ))

    # Update layout for the figure
    fig.update_layout(
        title=f'Boxplots of CO2 Equivalents Total for Different Treatments with {sample_size} Simulations',
        yaxis_title='CO2 Equivalents Total',
        xaxis_title='Treatment',
        boxmode='group'  # Group boxplots by treatment
    )

    # Show the plot
    fig.show()


def plot_violin_plots_plotly(sim_results_dict, result_key='co2_eq_tot', save_html=False, save_jpeg=False):
    # Create a Plotly figure
    fig = go.Figure()

    # Define a list of distinct colors for the violin plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Loop through each treatment in the simulation_results_dict to plot violin plots
    for i, (treatment, df_list) in enumerate(sim_results_dict.items()):
        # Map the treatment to the new name for plotting
        plot_name = pathway_name_mapping.get(treatment, treatment)  # Use the mapping, default to original name

        values = [dfc.find_value_in_results_df(df, result_key) for df in df_list]

        # Calculate IQR and determine cutoffs
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out the extreme outliers based on the IQR
        filtered_values = [x for x in values if lower_bound <= x <= upper_bound]

        # Recalculate statistics on the filtered values
        mean_value = np.mean(filtered_values)
        median_value = np.median(filtered_values)
        q1_filtered, q3_filtered = np.percentile(filtered_values, [25, 75])

        # Convert to thousands ('K') and format as string
        mean_value_str = f'{mean_value / 1000:.2f} K'
        median_value_str = f'{median_value / 1000:.2f} K'
        q1_filtered_str = f'{q1_filtered / 1000:.2f} K'
        q3_filtered_str = f'{q3_filtered / 1000:.2f} K'

        # Add the violin plot for the filtered values
        fig.add_trace(go.Violin(
            y=filtered_values,
            name=plot_name,
            line_color=colors[i % len(colors)],
            box_visible=True,
            meanline_visible=True
        ))

        """        # Add annotations for mean, median, Q1, Q3
        annotations = [
            {'x': plot_name, 'y': mean_value, 'text': f'Mean: {mean_value_str}', 'xshift': 20, 'yshift': 0},
            {'x': plot_name, 'y': median_value, 'text': f'Median: {median_value_str}', 'xshift': 20, 'yshift': 0},
            {'x': plot_name, 'y': q1_filtered, 'text': f'Q1: {q1_filtered_str}', 'xshift': 20, 'yshift': 0},
            {'x': plot_name, 'y': q3_filtered, 'text': f'Q3: {q3_filtered_str}', 'xshift': 20, 'yshift': 0}
        ]
        for ann in annotations:
            fig.add_annotation(x=ann['x'], y=ann['y'], text=ann['text'], showarrow=False,
                               xshift=ann['xshift'], yshift=ann['yshift'])"""

    # Update layout for the figure
    fig.update_layout(
        title=f'Violin Plots of {result_key} for Different Treatments',
        title_font_size=40,
        yaxis_title=f'{result_key}',
        yaxis_title_font_size=30,
        yaxis_tickfont_size=25,
        xaxis_title='Treatment',
        xaxis_title_font_size=30,
        xaxis_tickfont_size=25,
        violinmode='group'
    )

    # Show the plot
    fig.show()

    filename_plot = f'plots/violin_plots_{result_key}'
    # Save as HTML
    if save_html:
        html_filename = f"{filename_plot}.html"
        fig.write_html(html_filename)
        print(f"Plot saved as {html_filename}")

    # Save as JPEG
    if save_jpeg:
        jpeg_filename = f"{filename_plot}.jpeg"
        fig.write_image(jpeg_filename)
        print(f"Plot saved as {jpeg_filename}")


def plot_violin_plots_co2_per_kwh(sim_results_dict, result_key, save_html=False, save_jpeg=False):
    # Create a Plotly figure
    fig = go.Figure()

    # Define a list of distinct colors for the violin plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    co2_eq = 'co2_eq_tot'

    electricity = 'electricity_generated_tot'

    co2_per_kwh = result_key

    # Loop through each treatment in the simulation_results_dict to plot violin plots
    for i, (treatment, df_list) in enumerate(sim_results_dict.items()):
        # Map the treatment to the new name for plotting
        plot_name = pathway_name_mapping.get(treatment, treatment)  # Use the mapping, default to original name

        # Skip the iteration if treatment is 'no_treatment'
        if treatment == 'no_treatment':
            continue
        values = [dfc.find_value_in_results_df(df, co2_per_kwh) for df in df_list]

        # Calculate IQR and determine cutoffs
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out the extreme outliers based on the IQR
        filtered_values = [x for x in values if lower_bound <= x <= upper_bound]
        if not filtered_values:
            logging.warning(f'Filtered values for treatment "{treatment}" are empty after applying IQR filter.')

        # Recalculate statistics on the filtered values
        mean_value = np.mean(filtered_values)
        median_value = np.median(filtered_values)
        q1_filtered, q3_filtered = np.percentile(filtered_values, [25, 75])

        # Convert to thousands ('K') and format as string
        mean_value_str = f'{mean_value:.2f}'
        median_value_str = f'{median_value:.2f}'
        q1_filtered_str = f'{q1_filtered:.2f}'
        q3_filtered_str = f'{q3_filtered:.2f}'

        # Add the violin plot for the filtered values
        fig.add_trace(go.Violin(
            y=filtered_values,
            name=plot_name,
            line_color=colors[i % len(colors)],
            box_visible=True,
            meanline_visible=True
        ))

        """        # Instead of using 'i' as the x-coordinate, use the 'treatment' name
        annotations = [
            {'x': plot_name, 'y': mean_value, 'text': f'Mean: {mean_value_str}', 'xshift': 20, 'yshift': 0},
            {'x': plot_name, 'y': median_value, 'text': f'Median: {median_value_str}', 'xshift': 10, 'yshift': 0},
            {'x': plot_name, 'y': q1_filtered, 'text': f'Q1: {q1_filtered_str}', 'xshift': 20, 'yshift': 0},
            {'x': plot_name, 'y': q3_filtered, 'text': f'Q3: {q3_filtered_str}', 'xshift': 10, 'yshift': 0}
        ]
        for ann in annotations:
            fig.add_annotation(x=ann['x'], y=ann['y'], text=ann['text'], showarrow=False,
                               xshift=ann['xshift'], yshift=ann['yshift'], xanchor='center')"""

    # Update layout for the figure
    fig.update_layout(
        title=f'Violin Plots of {result_key} for Different Treatments',
        title_font_size=40,
        yaxis_title=f'kg CO2 eq. / kWh',
        yaxis_title_font_size=30,
        yaxis_tickfont_size=25,
        xaxis_title='Treatment',
        xaxis_title_font_size=30,
        xaxis_tickfont_size=25,
        violinmode='group'
    )

    # Show the plot
    fig.show()

    filename_plot = f'plots/violin_plots_{result_key}'
    # Save as HTML
    if save_html:
        html_filename = f"{filename_plot}.html"
        fig.write_html(html_filename)
        print(f"Plot saved as {html_filename}")

    # Save as JPEG
    if save_jpeg:
        jpeg_filename = f"{filename_plot}.jpeg"
        fig.write_image(jpeg_filename)
        print(f"Plot saved as {jpeg_filename}")


def plot_violin_plots_with_quotients_plotly(sim_results_dict, result_key='co2_eq_tot',
                                            comparison_scenario='no_treatment', save_html=False, save_jpeg=False):
    # Create a Plotly figure
    fig = go.Figure()

    # Extract the comparison data for the specified scenario
    comparison_data = [dfc.find_value_in_results_df(df, result_key) for df in
                       sim_results_dict.get(comparison_scenario, [])]
    comparison_mean = np.mean(comparison_data)

    # Define a list of distinct colors for the violin plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Loop through each scenario and compare it to the comparison_scenario
    for i, (scenario, df_list) in enumerate(sim_results_dict.items()):
        # Map the treatment to the new name for plotting
        plot_name = pathway_name_mapping.get(scenario, scenario)  # Use the mapping, default to original name
        # Map the treatment to the new name for plotting
        comparison_scenario_name = pathway_name_mapping.get(comparison_scenario, comparison_scenario)  # Use the mapping, default to original name

        if scenario != comparison_scenario:
            values = [dfc.find_value_in_results_df(df, result_key) / comparison_mean for df in df_list if
                      comparison_mean != 0]

            # Calculate IQR and determine cutoffs
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Filter out the extreme outliers based on the IQR
            filtered_values = [x for x in values if lower_bound <= x <= upper_bound]

            fig.add_trace(go.Violin(
                y=filtered_values,
                name=f"{plot_name}/{comparison_scenario_name}",
                line_color=colors[i % len(colors)],
                box_visible=True,
                meanline_visible=True
            ))

    # Add a horizontal line at Y=1
    fig.add_shape(
        type="line",
        x0=-0.5,  # Adjust x0 and x1 to ensure the line spans across the entire plot
        y0=1,
        x1=len(sim_results_dict.items()) - 0.5,
        y1=1,
        line=dict(
            color="Red",
            width=4,
            dash="dashdot",
        ),
    )

    # Update layout for the figure
    fig.update_layout(
        title=f'Violin Plots of Quotients Compared to {comparison_scenario}',
        title_font_size=40,
        yaxis_title='Q = Xi / Xcomparison',
        yaxis_title_font_size= 30,
        yaxis_tickfont_size=20,
        xaxis_title='Scenario',
        xaxis_title_font_size=30,
        xaxis_tickfont_size=20,
        violinmode='group',
        showlegend=False
    )

    # Show the plot
    fig.show()

    filename_plot = f'plots/violin_plots_quotients_{result_key}_{comparison_scenario}'
    # Save as HTML
    if save_html:
        html_filename = f"{filename_plot}.html"
        fig.write_html(html_filename)
        print(f"Plot saved as {html_filename}")

    # Save as JPEG
    if save_jpeg:
        jpeg_filename = f"{filename_plot}.jpeg"
        fig.write_image(jpeg_filename)
        print(f"Plot saved as {jpeg_filename}")


def plot_source_contributions_violin(sim_results_dict, source_columns):
    import plotly.graph_objects as go

    for scenario, df_list in sim_results_dict.items():
        # Create a new figure for each scenario
        fig = go.Figure()

        for source in source_columns:
            # Extract data for each source
            source_data = [dfc.find_value_in_results_df(df, source) for df in df_list]

            # Add a violin plot for this source
            fig.add_trace(go.Violin(
                y=source_data,
                x=[source] * len(source_data),
                name=f"{source}",
                box_visible=True,
                meanline_visible=True
            ))

        # Update layout for each scenario's figure
        fig.update_layout(
            title=f'Source Contributions to CO2 Equivalent in {scenario} Scenario',
            yaxis_title='Contribution',
            xaxis_title='Source',
            violinmode='group',
            showlegend=False
        )

        # Show the plot
        fig.show()


def plot_relative_source_contributions_violin(sim_results_dict, source_columns, total_impact_key, save_html=False, save_jpeg=False):
    # Define a color palette
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                     '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#17becf', '#9edae5']

    # Create a color mapping for each source
    color_mapping = {source: color_palette[i % len(color_palette)] for i, source in enumerate(source_columns)}

    for scenario, df_list in sim_results_dict.items():
        # Map the treatment to the new name for plotting
        plot_name = pathway_name_mapping.get(scenario, scenario)  # Use the mapping, default to original name

        # Create a new figure for each scenario
        fig = go.Figure()

        # Initialize a dictionary to collect all contributions for each source
        all_contributions = {source: [] for source in source_columns}

        for df in df_list:
            # Extract total impact for the current dataframe
            total_impact = dfc.find_value_in_results_df(df, total_impact_key)

            # Skip if total impact is zero to avoid division by zero
            if total_impact == 0:
                continue

            # Calculate relative contributions for each source
            for source in source_columns:
                source_value = dfc.find_value_in_results_df(df, source)
                relative_contribution = 100 * (source_value / total_impact)
                all_contributions[source].append(relative_contribution)

        # Filter out sources with only zero contributions before plotting
        for source in list(all_contributions):  # Use list to make a copy of keys to iterate over
            contributions = all_contributions[source]
            if not any(contributions):  # If there are no non-zero contributions, remove the source
                del all_contributions[source]

        # Add violin plots to the figure for each source
        for source, contributions in all_contributions.items():
            if contributions:  # Check if there are contributions to plot
                # Calculate Interquartile Range (IQR) and determine cutoffs, cutoff set at 1.5 * IQR
                q1, q3 = np.percentile(contributions, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Filter out the extreme outliers
                filtered_contributions = [x for x in contributions if lower_bound <= x <= upper_bound]

                fig.add_trace(go.Violin(
                    y=filtered_contributions,
                    # Use the mapped name for the x-axis
                    x=[source_name_mapping[source]] * len(filtered_contributions),
                    name=source_name_mapping[source],  # Also use the mapped name for the legend
                    line_color=color_mapping[source],
                    box_visible=True,
                    meanline_visible=True
                ))

        # Update layout for each scenario's figure
        fig.update_layout(
            title=f'Relative Source Contributions to Total {total_impact_key} in {plot_name}',
            title_font_size= 40,
            yaxis_title='Contribution (%)',
            yaxis_title_font_size=30,
            yaxis_tickfont_size=20,
            xaxis_title='Source',
            xaxis_title_font_size=30,
            xaxis_tickfont_size=20,
            violinmode='group',
            showlegend=False
        )

        # Show the plot
        fig.show()

        filename_plot = f'plots/violin_plots_relative_source_contribution_{total_impact_key}_{scenario}'
        # Save as HTML
        if save_html:
            html_filename = f"{filename_plot}.html"
            fig.write_html(html_filename)
            print(f"Plot saved as {html_filename}")

        # Save as JPEG
        if save_jpeg:
            jpeg_filename = f"{filename_plot}.jpeg"
            fig.write_image(jpeg_filename)
            print(f"Plot saved as {jpeg_filename}")



# list with sources of CO2 eq. emissions for source contribution
# Original source names
co2_sources_list = ["co2_methane_pre_storage", "co2_methane_post_storage", "co2_methane_ad",
                    "co2_methane_biogas_upgrading", "co2_n2o_pre_storage", "co2_n2o_post_storage",
                    "co2_n2o_field", "co2_electricity_demand_ad", "co2_electricity_demand_biogas_upgrading",
                    "co2_transport", "co2_ad_construction", "co2_chp_construction"]

# New names for plotting
co2_sources_list_plot_names = ["CH4 Pre-Storage", "CH4 Post-Storage", "CH4 AD losses",
                                "CH4 biogas upgrading losses", "N2O Pre-Storage", "N2O Post-Storage",
                                "N2O Field", "Electricity Demand AD", "Electricity Demand Biogas Upgrading",
                                "Transport", "AD plant construction", "CHP plant construction"]

# Create a mapping between the original source names and the new plot names
source_name_mapping = {original: new for original, new in zip(co2_sources_list, co2_sources_list_plot_names)}

#list with pathway-names and their mapping for plots
# Original names in code
pathway_names_code = ['no_treatment', 'ad_only', 'ad_biogas', 'steam_ad', 'steam_ad_biogas']

# New names for plotting
pathway_names_plot = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

# Create a mapping between the original source names and the new plot names
pathway_name_mapping = {original: new for original, new in zip(pathway_names_code, pathway_names_plot)}


if __name__ == "__main__":

    """# Read the configuration files
    animal_config = dfc.read_file_to_dataframe(default_animal_config_path)
    env_config = dfc.read_file_to_dataframe(default_environmental_config_path)

    # Identify parameters with non-zero standard deviation in both configs
    stdev_cols_animal = [col for col in animal_config.columns if "stdev" in col]
    value_cols_animal = [col.replace("stdev_", "") for col in stdev_cols_animal]
    animal_parameters_to_sample = animal_config[animal_config[stdev_cols_animal].sum(axis=1) != 0]
    env_parameters_to_sample = env_config[env_config['stdev'] != 0]"""

    # Specify the file path
    simulation_results_file_path = 'monte_carlo_simulation/simulation_results.h5'
    sim_results_dict = {}
    # Check if the file exists
    if os.path.isfile(simulation_results_file_path):
        print("File 'simulation_results.h5' exists. Skipping Monte Carlo simulation.")
        sim_results_dict = load_from_hdf5(simulation_results_file_path)
        print('Loading of the File complete, starting visualization.')

    else:
        print("File 'simulation_results.h5' does not exist. Running Monte Carlo simulation.")
        # Read the combined configuration file
        combined_config = dfc.read_file_to_dataframe(combined_config_path)



        # Splitting the combined configuration
        default_animal_config = dfc.read_file_to_dataframe(default_animal_config_path)
        default_env_config = dfc.read_file_to_dataframe(default_environmental_config_path)

        env_config_df, animal_config_df = split_combined_config(combined_config)
        animal_config_df = transform_animal_config(animal_config_df)

        # For Animal Configuration
        # We want to make sure that both mu and sigma are non-zero for a parameter.
        animal_parameters_to_sample = animal_config_df.loc[
            (animal_config_df.filter(regex='_mu$').ne(0) & animal_config_df.filter(regex='_sigma$').ne(0)).all(axis=1)
        ]

        # For Environmental Configuration
        env_parameters_to_sample = env_config_df.loc[
            ~env_config_df['Distribution function'].str.lower().eq('none') & (
                    (env_config_df['Distribution function'].str.lower().eq('triangle') &
                     env_config_df[['upper', 'lower', 'median']].notna().all(axis=1)) |
                    (env_config_df['Distribution function'].str.lower().eq('lognormal') &
                     env_config_df[['sigma', 'mu']].notna().all(axis=1))
            )
            ]

        # Create a dictionary for fast retrieval of mu and sigma values
        mu_sigma_dict = {}
        for _, row in animal_config_df.iterrows():
            animal_type = row['animal_type']
            stable_type = row['stable_type']
            for col in animal_config_df.columns:
                if col.endswith("_mu") or col.endswith("_sigma"):
                    param_base = col.rsplit("_", 1)[0]  # Extract base parameter name
                    key = (animal_type, stable_type, param_base)
                    mu_sigma_dict[key] = (row[f"{param_base}_mu"], row[f"{param_base}_sigma"])

        # Create a list to store samples for animal configuration
        animal_samples = []

        # Iterate over each row in animal_config
        for index, row in animal_config_df.iterrows():
            for iteration in range(sample_size):
                sample_data = {'animal_type': row['animal_type'], 'stable_type': row['stable_type'],
                               'iteration': iteration}

                # Iterate over each parameter in animal_config
                for param_base in set(
                        col.rsplit("_", 1)[0] for col in animal_config_df.columns if "_mu" in col or "_sigma" in col):
                    key = (row['animal_type'], row['stable_type'], param_base)
                    if key in mu_sigma_dict:
                        mu_value, sigma_value = mu_sigma_dict[key]
                        if mu_value and sigma_value:
                            sample_value = generate_lognormal_mu_sigma(mu_value, sigma_value, 1)[0]
                            sample_data[param_base] = sample_value

                animal_samples.append(sample_data)

        # Generate samples for environmental parameters
        env_samples = {}
        for _, row in env_parameters_to_sample.iterrows():
            param_name = row['name']
            dist_function = row['Distribution function'].lower()
            if dist_function == 'lognormal':
                samples = generate_lognormal_mu_sigma(row['mu'], row['sigma'], sample_size)
            elif dist_function == 'triangle':
                samples = generate_triangle(row['lower'], row['upper'], row['median'], sample_size)
            env_samples[param_name] = samples

            # Assume farm_data_df_list is a list of dataframes, each representing a farm file
        farm_data_dict = {file_name: dfc.read_file_to_dataframe(os.path.join(farm_files_path, file_name))
                          for file_name in farm_paths_list}

        # Generate random samples for each farm file
        farm_samples_dict = {}
        for farm_file_name, farm_df in farm_data_dict.items():
            uncertainty_params = extract_uncertainty_parameters(farm_df)
            farm_samples = generate_random_values_for_farm(uncertainty_params, sample_size)
            farm_samples_dict[farm_file_name] = farm_samples

        # Convert the samples dictionaries to DataFrames
        df_env_samples = pd.DataFrame(env_samples)
        df_animal_samples = pd.DataFrame(animal_samples)

        # Save the samples to Excel files
        df_animal_samples.to_excel("monte_carlo_simulation/saved_animal_samples.xlsx", index=False)
        df_env_samples.to_excel("monte_carlo_simulation/saved_env_samples.xlsx", index=False)






        sim_results_dict = create_config_for_simulation(df_animal_samples, df_env_samples, default_env_config, default_animal_config, sample_size, farm_data_df_list, farm_samples_dict)

    """write_dict_to_csv(sim_results_dict, 'Debug/no_hdf5.csv')
    sim_results_dict_test = load_from_hdf5()
    write_dict_to_csv(sim_results_dict_test, 'Debug/hdf5.csv')"""


    plot_violin_plots_co2_per_kwh(sim_results_dict, 'co2_per_kwh_el', save_html=True, save_jpeg=False)
    plot_violin_plots_co2_per_kwh(sim_results_dict, 'co2_per_kwh_energy', save_html=True, save_jpeg=False)
    plot_violin_plots_plotly(sim_results_dict, save_html=True, save_jpeg=False)
    plot_violin_plots_plotly(sim_results_dict, "daly_nh3", save_html=True, save_jpeg=False)
    plot_violin_plots_plotly(sim_results_dict, "co2_electricity_mix", save_html=True, save_jpeg=False)
    plot_violin_plots_with_quotients_plotly(sim_results_dict, save_html=True, save_jpeg=False)
    plot_violin_plots_with_quotients_plotly(sim_results_dict, comparison_scenario="ad_only", save_html=True, save_jpeg=False)
    plot_violin_plots_with_quotients_plotly(sim_results_dict, comparison_scenario="ad_biogas", save_html=True, save_jpeg=False)
    plot_violin_plots_with_quotients_plotly(sim_results_dict, comparison_scenario="steam_ad", save_html=True, save_jpeg=False)
    plot_violin_plots_with_quotients_plotly(sim_results_dict, comparison_scenario="steam_ad_biogas", save_html=True, save_jpeg=False)
    plot_relative_source_contributions_violin(sim_results_dict, co2_sources_list, "co2_eq_tot", save_html=True, save_jpeg=False)







