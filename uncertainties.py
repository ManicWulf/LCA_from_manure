import pandas as pd
import numpy as np
import os
import time
import plotly.graph_objects as go
import logging



import farm_calc as fc
import dash_functions_and_callbacks as dfc


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',  # This will log to a file named app.log. Remove this to log to console.
                    filemode='w')  # This means the log file will be overwritten each time the app is started. Use 'a' to append.


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

"""



Change the value for sample_size to adjust the number of simulation runs




"""
# Define the sample_size variable
sample_size = 100  # You can adjust this value as needed


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



def load_from_hdf5(filename="simulation_results.h5"):
    results = {}
    filepath = f'monte_carlo_simulation/{filename}'
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
    if list_farms is None:
        list_farms = farm_data_df_list
    farm_animal_calc = fc.calc_all_farms_new(list_farms, animal_config, env_config)

    # 2. Perform the no-treatment calculations using farm_animal_calc
    no_treatment_df = fc.post_storage_and_field_emissions(farm_animal_calc, env_config, False)
    no_treatment_df = fc.calc_env_impacts(no_treatment_df, env_config)

    # 3. Perform AD only calculations
    ad_only_df = fc.calc_anaerobic_digestion(farm_animal_calc, env_config)
    ad_only_df = fc.post_storage_and_field_emissions(ad_only_df, env_config, True)
    ad_only_df = fc.calc_chp_output(ad_only_df, env_config)
    ad_only_df = fc.calc_env_impacts(ad_only_df, env_config)

    # 4. Perform biogas upgrading calculations for AD+Biogas based on ad_only_df
    ad_biogas_df = fc.calc_biogas_upgrading(ad_only_df, env_config)
    ad_biogas_df = fc.calc_env_impacts(ad_biogas_df, env_config)

    # 5. Calculate steam pretreatment with AD based on farm_animal_calc
    steam_ad_df = fc.steam_pre_treatment(farm_animal_calc, env_config)
    steam_ad_df = fc.calc_anaerobic_digestion(steam_ad_df, env_config)
    steam_ad_df = fc.post_storage_and_field_emissions(steam_ad_df, env_config, True)
    steam_ad_df = fc.calc_chp_output(steam_ad_df, env_config)
    steam_ad_df = fc.calc_env_impacts(steam_ad_df, env_config)

    # 6. Calculate biogas upgrading for steam_ad_df
    steam_ad_biogas_df = fc.calc_biogas_upgrading(steam_ad_df, env_config)
    steam_ad_biogas_df = fc.calc_env_impacts(steam_ad_biogas_df, env_config)

    logging.debug(f'Steam AD biogas dataframe for uncertainties: {steam_ad_biogas_df}')
    logging.debug(f'no treatment dataframe for uncertainties: {no_treatment_df}')
    logging.debug((f'Env config: {env_config}'))
    logging.debug(f'animal config: {animal_config}')

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
def generate_lognormal(mean, std, size):
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


def create_config_for_simulation(df_animal_samples, df_env_samples, env_config, num_simulations):
    simulation_results_dict = {}
    for n in range(num_simulations):
        # Select one sample per unique combination of animal type and manure type
        animal_config_df = df_animal_samples.drop_duplicates(subset=['animal_type', 'stable_type'])

        # If similar logic is needed for environmental samples, do it here
        env_config_df = substitute_env_config_values(df_env_samples, env_config, n)

        # Use config_df in your model here

        simulation_results_dict = create_results_list_monte_carlo(simulation_results_dict, *lca_calculation(env_config_df, animal_config_df))

        # Optionally, save config_df to check its structure
        #animal_config_df.to_excel(f"Monte carlo simulation/animal_config_test{n}.xlsx", index=False)
        #env_config_df.to_excel(f"Monte carlo simulation/env_config_test{n}.xlsx", index=False)

    save_to_hdf5(simulation_results_dict)

    return simulation_results_dict


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

if __name__ == "__main__":

    # Read the configuration files
    animal_config = dfc.read_file_to_dataframe(default_animal_config_path)
    env_config = dfc.read_file_to_dataframe(default_environmental_config_path)

    # Identify parameters with non-zero standard deviation in both configs
    stdev_cols_animal = [col for col in animal_config.columns if "stdev" in col]
    value_cols_animal = [col.replace("stdev_", "") for col in stdev_cols_animal]
    animal_parameters_to_sample = animal_config[animal_config[stdev_cols_animal].sum(axis=1) != 0]
    env_parameters_to_sample = env_config[env_config['stdev'] != 0]


    # Create a list to store rows of the new DataFrame
    all_samples = []

    # Iterate over each row in animal_config
    for index, row in animal_config.iterrows():
        # For each set of samples
        for _ in range(sample_size):
            sample_data = {
                'animal_type': row['animal_type'],
                'stable_type': row['stable_type']
            }
            # For each stdev and value column pair, generate a sample and add it to the sample_data dictionary
            for stdev_col, value_col in zip(stdev_cols_animal, value_cols_animal):
                sample_data[value_col] = generate_lognormal(row[value_col], row[stdev_col], 1)[0]
            # Add the sample_data dictionary to the all_samples list
            all_samples.append(sample_data)

    # Generate samples for environmental parameters
    env_samples = {}
    for _, row in env_parameters_to_sample.iterrows():
        param_name = row['name']
        samples = generate_lognormal(row['value'], row['stdev'], sample_size)
        env_samples[param_name] = samples

    # Convert the samples dictionaries to DataFrames
    df_env_samples = pd.DataFrame(env_samples)
    df_animal_samples = pd.DataFrame(all_samples)

    # Save the samples to Excel files
    df_animal_samples.to_excel("monte_carlo_simulation/saved_animal_samples.xlsx", index=False)
    df_env_samples.to_excel("monte_carlo_simulation/saved_env_samples.xlsx", index=False)






    simulation_results_dict = create_config_for_simulation(df_animal_samples, df_env_samples, env_config, sample_size)
    sim_results_dict_test = load_from_hdf5()
    write_dict_to_csv(simulation_results_dict, 'Debug/no_hdf5.csv')
    write_dict_to_csv(sim_results_dict_test, 'Debug/hdf5.csv')
    plot_histograms_plotly(sim_results_dict_test)








