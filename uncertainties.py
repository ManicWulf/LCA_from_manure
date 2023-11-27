import pandas as pd
import numpy as np
import dash_functions_and_callbacks as dfc

"""
dash.register_page(__name__)

layout = html.Div([
    html.H1('This is our Uncertainties page'),
    html.Div('This is our Uncertainties content.'),
])
"""
# define path to animal and environmental config
default_animal_config_path = "default_configs/default_animal_config.xlsx"
default_environmental_config_path = "default_configs/default_environmental_config.xlsx"

# define path to farm files folder
farm_files_path = "farm files/"


# Define the sample_size variable
sample_size = 10  # You can adjust this value as needed

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



def substitute_env_config_values(samples_df, n):
    """
    Substitute values in the default environmental configuration with values from a sample row.

    :param samples_df: The generated samples dataframe.
    :param n: Index of the row in samples_df to use for substitution.
    :return: A new dataframe with substituted values.
    """
    # Create a copy of the default configuration to prevent in-place modification
    changed_config = env_config.copy()

    # Iterate over each parameter in the default config
    for index, row in changed_config.iterrows():
        # If stdev is not 0 and not NaN
        if row['stdev'] != 0 and not pd.isna(row['stdev']):
            # Substitute the value in the copied dataframe with value from the sample
            param_name = row['name']
            changed_config.at[index, 'value'] = generate_lognormal(row['value'], row['stdev'], n)

    return changed_config





def substitute_animal_config_values(samples_df, num_simulations):
    """
    Substitute values in the default animal configuration dataframe with values from a sample row based on the given index.

    :param samples_df: The generated animal samples dataframe.
    :param n: The row index to be used from the samples_df for substitution.
    :return: A new dataframe with substituted values.
    """
    # Preparing the DataFrame for Monte Carlo simulation
    simulation_df = pd.DataFrame()

    # Iterate over each row in the original DataFrame
    for index, row in samples_df.iterrows():
        # Dictionary to store one simulated row
        simulated_row = {}

        # Iterate over each column
        for col in samples_df.columns:
            # Check if this column has a corresponding standard deviation column
            if 'stdev_' + col in samples_df.columns:
                mean = row[col]
                std = row['stdev_' + col]
                simulated_row[col] = generate_lognormal(mean, std, num_simulations)
            else:
                # For columns without standard deviation, repeat the value
                simulated_row[col] = np.full(num_simulations, row[col])

        # Convert the dictionary to a DataFrame and append it to the main DataFrame
        simulated_row_df = pd.DataFrame(simulated_row)
        simulation_df = pd.concat([simulation_df, simulated_row_df], ignore_index=True)

    return simulation_df



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
    df_animal_samples.to_excel("Monte carlo simulation/saved_animal_samples.xlsx", index=False)
    df_env_samples.to_excel("Monte carlo simulation/saved_env_samples.xlsx", index=False)





    for n in range(sample_size):
        new_config_env = substitute_env_config_values(df_env_samples, n)
        #new_config_env.to_excel(f"Monte carlo simulation/new_env_config_test{n}.xlsx", index=False)
        new_config_animal = substitute_animal_config_values(df_animal_samples, n)
        new_config_animal.to_excel(f"Monte carlo simulation/new_animal_config_test{n}.xlsx", index=False)







