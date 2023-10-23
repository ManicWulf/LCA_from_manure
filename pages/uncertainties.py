import dash
from dash import html
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


# Define the sample_size variable
sample_size = 10  # You can adjust this value as needed


# Function to generate samples from a lognormal distribution
def generate_lognormal_samples(mean, stdev, sample_size):
    if mean == 0:
        return np.zeros(sample_size)
    sigma = np.sqrt(np.log(1 + (stdev / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)
    return samples


if __name__ == "__main__":

    # Read the configuration files
    animal_config = dfc.read_file_to_dataframe(default_animal_config_path)
    env_config = dfc.read_file_to_dataframe(default_environmental_config_path)

    # Identify parameters with non-zero standard deviation in both configs
    stdev_cols_animal = [col for col in animal_config.columns if "stdev" in col]
    value_cols_animal = [col.replace("stdev_", "") for col in stdev_cols_animal]
    animal_parameters_to_sample = animal_config[animal_config[stdev_cols_animal].sum(axis=1) != 0]
    env_parameters_to_sample = env_config[env_config['stdev'] != 0]

    # Generate samples for animal parameters
    animal_samples = {}
    for stdev_col, value_col in zip(stdev_cols_animal, value_cols_animal):
        for _, row in animal_parameters_to_sample.iterrows():
            param_name = row['animal_type'] + "_" + value_col
            samples = generate_lognormal_samples(row[value_col], row[stdev_col], sample_size)
            animal_samples[param_name] = samples

    # Generate samples for environmental parameters
    env_samples = {}
    for _, row in env_parameters_to_sample.iterrows():
        param_name = row['name']
        samples = generate_lognormal_samples(row['value'], row['stdev'], sample_size)
        env_samples[param_name] = samples

    # Convert the samples dictionaries to DataFrames
    df_animal_samples = pd.DataFrame(animal_samples)
    df_env_samples = pd.DataFrame(env_samples)

    # Save the samples to Excel files
    df_animal_samples.to_excel("/path/to/saved_animal_samples.xlsx", index=False)
    df_env_samples.to_excel("/path/to/saved_env_samples.xlsx", index=False)















