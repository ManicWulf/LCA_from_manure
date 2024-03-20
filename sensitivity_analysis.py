import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import dash_functions_and_callbacks as dfc
import uncertainties

def load_excel_data(file_path):
    """
    Load data from an Excel file.

    Args:
    file_path (str): The path to the Excel file.

    Returns:
    DataFrame: A pandas DataFrame containing the data from the Excel file.
    """
    return pd.read_excel(file_path)


def extract_unique_animals(dataframe, animal_type_column):
    """
    Extracts unique animal types from the DataFrame.

    Args:
    dataframe (DataFrame): The DataFrame to process.
    animal_type_column (str): The column name in the DataFrame that contains animal types.

    Returns:
    list: A list of unique animal types.
    """
    return dataframe[animal_type_column].unique().tolist()

def load_data_from_directory(directory_path):
    """
    Load all Excel files from a specified directory into DataFrames.

    Args:
    directory_path (str): The path to the directory containing Excel files.

    Returns:
    list of DataFrame: A list of pandas DataFrames loaded from each Excel file in the directory.
    """
    dataframes = []
    for file in glob.glob(os.path.join(directory_path, '*.xlsx')):
        df = pd.read_excel(file)
        dataframes.append(df)
    return dataframes

def aggregate_data(dataframes, unique_animals, aggregation_columns):
    """
    Aggregates data based on animal types and other criteria.

    Args:
    dataframes (list of DataFrame): List of DataFrames to be aggregated.
    unique_animals (list): List of unique animal types.
    aggregation_columns (list of str): Column names to aggregate data on.

    Returns:
    DataFrame: A pandas DataFrame after aggregation.
    """
    aggregated_data = pd.DataFrame()
    for animal in unique_animals:
        for df in dataframes:
            aggregated_animal_data = df[df['AnimalType'] == animal].groupby(aggregation_columns).sum()
            aggregated_data = pd.concat([aggregated_data, aggregated_animal_data], axis=0)
    return aggregated_data


def replace_nan_with_zero(dataframe):
    """
    Replace all NaN values in the DataFrame with 0.

    Args:
    dataframe (DataFrame): The DataFrame to process.

    Returns:
    DataFrame: The DataFrame with NaN values replaced by 0.
    """
    return dataframe.fillna(0)


def remove_outliers(df, column):
    """
    Remove outliers from a dataframe based on the IQR method.

    Args:
    df (DataFrame): The DataFrame to process.
    column (str): The name of the column to check for outliers.

    Returns:
    DataFrame: A DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df


def single_bootstrap_iteration(X, y):
    """
    Perform a single bootstrap iteration for calculating feature importances.

    Args:
    X (DataFrame): Features dataset.
    y (Series): Target variable.

    Returns:
    ndarray: Feature importances from the iteration.
    """
    X_sample, y_sample = resample(X, y)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_sample, y_sample)
    return model.feature_importances_


def calculate_feature_importances_parallel(X, y, num_bootstraps=1000, n_jobs=-1):
    """
    Calculate feature importances using Gradient Boosting and Bootstrapping with parallel processing.

    Args:
    X (DataFrame): Features dataset.
    y (Series): Target variable.
    num_bootstraps (int): Number of bootstrap iterations.
    n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.

    Returns:
    tuple: Mean feature importances, lower bounds, and upper bounds of confidence intervals.
    """
    bootstrap_importances = Parallel(n_jobs=n_jobs)(
        delayed(single_bootstrap_iteration)(X, y) for _ in range(num_bootstraps))

    # Calculate confidence intervals
    lower_bounds = np.percentile(bootstrap_importances, 2.5, axis=0)
    upper_bounds = np.percentile(bootstrap_importances, 97.5, axis=0)
    return np.mean(bootstrap_importances, axis=0), lower_bounds, upper_bounds


def plot_feature_importances(features, importances, ci_lower, ci_upper):
    """
    Plot feature importances with confidence intervals.

    Args:
    features (list): List of feature names.
    importances (list): List of feature importances.
    ci_lower (list): Lower bounds of the confidence intervals.
    ci_upper (list): Upper bounds of the confidence intervals.
    """
    # Create a scatter plot
    plt.scatter(features, importances, color='blue')

    # Add error bars for confidence intervals
    plt.errorbar(features, importances, yerr=[ci_lower, ci_upper], fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0)

    # Improve plot aesthetics
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances with Confidence Intervals')
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_filtered_feature_importances(features, importances, ci_lower, ci_upper, threshold=0.05):
    """
    Plot filtered feature importances with confidence intervals.

    Args:
    features (list): List of feature names.
    importances (list): List of feature importances.
    ci_lower (list): Lower bounds of the confidence intervals.
    ci_upper (list): Upper bounds of the confidence intervals.
    threshold (float): Minimum importance value to include in the plot.
    """
    # Filter out features with importance below the threshold
    filtered_features = []
    filtered_importances = []
    filtered_ci_lower = []
    filtered_ci_upper = []

    for feature, importance, lower, upper in zip(features, importances, ci_lower, ci_upper):
        if importance >= threshold:
            filtered_features.append(feature)
            filtered_importances.append(importance)
            filtered_ci_lower.append(lower)
            filtered_ci_upper.append(upper)

    # Create a scatter plot with the filtered data
    plt.scatter(filtered_features, filtered_importances, color='blue')

    # Add error bars for confidence intervals
    plt.errorbar(filtered_features, filtered_importances, yerr=[filtered_ci_lower, filtered_ci_upper], fmt='o',
                 color='red', ecolor='lightgray', elinewidth=3, capsize=0)

    # Improve plot aesthetics
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Filtered Feature Importances with Confidence Intervals')
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_importances(features, scaled_importances, unscaled_importances, scaled_ci_lower, scaled_ci_upper, unscaled_ci_lower, unscaled_ci_upper, threshold=0.05):
    """
    Plot both scaled and unscaled feature importances for features exceeding a threshold in scaled importances, along with their respective confidence intervals.

    Args:
    features (list): List of feature names.
    scaled_importances (list): Scaled feature importances.
    unscaled_importances (list): Original, unscaled feature importances.
    scaled_ci_lower (list): Lower bounds of the confidence intervals for scaled importances.
    scaled_ci_upper (list): Upper bounds of the confidence intervals for scaled importances.
    unscaled_ci_lower (list): Lower bounds of the confidence intervals for unscaled importances.
    unscaled_ci_upper (list): Upper bounds of the confidence intervals for unscaled importances.
    threshold (float): Threshold for scaled importances to decide which features to plot.
    """

    # Convert features to list if it's a pandas Index
    if isinstance(features, pd.Index):
        features = features.tolist()

    # Filter features based on scaled importances
    filtered_features = [feat for feat, imp in zip(features, scaled_importances) if imp >= threshold]
    filtered_scaled_importances = [imp for imp in scaled_importances if imp >= threshold]
    filtered_unscaled_importances = [unscaled_importances[features.index(feat)] for feat in filtered_features]
    filtered_scaled_ci_lower = [scaled_ci_lower[features.index(feat)] for feat in filtered_features]
    filtered_scaled_ci_upper = [scaled_ci_upper[features.index(feat)] for feat in filtered_features]
    filtered_unscaled_ci_lower = [unscaled_ci_lower[features.index(feat)] for feat in filtered_features]
    filtered_unscaled_ci_upper = [unscaled_ci_upper[features.index(feat)] for feat in filtered_features]

    # Filter features and calculate yerr for error bars
    yerr_scaled = [[imp - lower, upper - imp] for imp, lower, upper in zip(filtered_scaled_importances, filtered_scaled_ci_lower, filtered_scaled_ci_upper)]
    yerr_unscaled = [[imp - lower, upper - imp] for imp, lower, upper in zip(filtered_unscaled_importances, filtered_unscaled_ci_lower, filtered_unscaled_ci_upper)]

    yerr_scaled = np.array(yerr_scaled).T  # Transpose to match the shape expected by plt.errorbar
    yerr_unscaled = np.array(yerr_unscaled).T

    # Plotting scaled importances
    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_features, filtered_scaled_importances, color='blue')
    plt.errorbar(filtered_features, filtered_scaled_importances, yerr=yerr_scaled, fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.xticks(rotation=45, ha='right')
    plt.title('Scaled Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Scaled Importance')
    plt.tight_layout()
    plt.show()

    # Plotting unscaled importances
    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_features, filtered_unscaled_importances, color='green')
    plt.errorbar(filtered_features, filtered_unscaled_importances, yerr=yerr_unscaled, fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.xticks(rotation=45, ha='right')
    plt.title('Unscaled Feature Importances of Filtered Features')
    plt.xlabel('Features')
    plt.ylabel('Unscaled Importance')
    plt.tight_layout()
    plt.show()


def scale_importances_and_cis_by_max(importances, ci_lower, ci_upper):
    """
    Scale feature importances and their confidence intervals based on the maximum importance value.

    Args:
    importances (list or numpy array): List or array of feature importances.
    ci_lower (list or numpy array): Lower bounds of the confidence intervals.
    ci_upper (list or numpy array): Upper bounds of the confidence intervals.

    Returns:
    tuple: Scaled feature importances, scaled lower bounds, and scaled upper bounds.
    """
    max_importance = np.max(importances)
    scaled_importances = importances / max_importance
    scaled_ci_lower = ci_lower / max_importance
    scaled_ci_upper = ci_upper / max_importance
    return scaled_importances, scaled_ci_lower, scaled_ci_upper


def save_feature_importances_to_hdf5(features, importances, ci_lower, ci_upper, filename="feature_importances.h5"):
    filepath = f'monte_carlo_simulation/{filename}'
    data = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })

    with pd.HDFStore(filepath, mode='w') as store:
        store.put('feature_importances', data, format='table')


def load_feature_importances_from_hdf5(filepath='monte_carlo_simulation/feature_importances.h5'):
    with pd.HDFStore(filepath, mode='r') as store:
        feature_importances = store.get('feature_importances')
    return feature_importances


animal_config_samples_path = "monte_carlo_simulation/saved_animal_samples.xlsx"
environmental_config_samples_path = "monte_carlo_simulation/saved_env_samples.xlsx"
sim_results_path = "monte_carlo_simulation/simulation_results.h5"
farm_files_folder_path = 'farm files'

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

# Replace NaN values with 0
combined_samples = replace_nan_with_zero(combined_samples)


# Display the complete dataframe for verification
print(combined_samples)

# Load simulation results
sim_results_dict = uncertainties.load_from_hdf5(sim_results_path)

# Prepare the dataset for each treatment
result_key = 'co2_eq_tot'  # replace with your actual result key
treatment_datasets = {}

for treatment, df_list in sim_results_dict.items():
    treatment_results = [dfc.find_value_in_results_df(df, result_key) for df in df_list]

    # Convert the treatment_results list to a DataFrame
    treatment_results_df = pd.DataFrame(treatment_results, columns=[result_key])

    # Remove outliers from the treatment_results DataFrame
    treatment_results_df = remove_outliers(treatment_results_df, result_key)

    # Ensure combined_samples is synchronized with treatment_results_df
    treatment_datasets[treatment] = combined_samples.loc[treatment_results_df.index].copy()
    treatment_datasets[treatment]['results'] = treatment_results_df[result_key]


# Define a function for calculating feature importances using Gradient Boosting and Bootstrapping
def calculate_feature_importances(X, y, num_bootstraps=1000):
    bootstrap_importances = []
    for _ in range(num_bootstraps):
        X_sample, y_sample = resample(X, y)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_sample, y_sample)
        bootstrap_importances.append(model.feature_importances_)

    # Calculate confidence intervals
    lower_bounds = np.percentile(bootstrap_importances, 2.5, axis=0)
    upper_bounds = np.percentile(bootstrap_importances, 97.5, axis=0)
    return np.mean(bootstrap_importances, axis=0), lower_bounds, upper_bounds


# Example of training a model for one treatment
# treatment = list(treatment_datasets.keys())[0]  # Replace with the specific treatment you want to analyze
treatment = 'ad_only'
X = treatment_datasets[treatment].drop(columns=['results'])
y = treatment_datasets[treatment]['results']

# Define a function for calculating feature importances using Gradient Boosting and Bootstrapping
def calculate_feature_importances(X, y, num_bootstraps=1000):
    bootstrap_importances = []
    for _ in range(num_bootstraps):
        X_sample, y_sample = resample(X, y)
        model = GradientBoostingRegressor(n_estimators=1000, random_state=42)
        model.fit(X_sample, y_sample)
        bootstrap_importances.append(model.feature_importances_)

    # Calculate confidence intervals
    lower_bounds = np.percentile(bootstrap_importances, 2.5, axis=0)
    upper_bounds = np.percentile(bootstrap_importances, 97.5, axis=0)
    return np.mean(bootstrap_importances, axis=0), lower_bounds, upper_bounds


filepath_features = 'monte_carlo_simulation/feature_importances.h5'

if os.path.exists(filepath_features):
    # Load the stored feature importances and confidence intervals
    importances_data = load_feature_importances_from_hdf5(filepath_features)
    feature_importances = importances_data['Importance']
    ci_lower = importances_data['CI_Lower']
    ci_upper = importances_data['CI_Upper']
else:
    # Calculate feature importances and confidence intervals
    feature_importances, ci_lower, ci_upper = calculate_feature_importances_parallel(X, y)
    save_feature_importances_to_hdf5(X.columns, feature_importances, ci_lower, ci_upper)

feature_importances_scaled, ci_lower_scaled, ci_upper_scaled = scale_importances_and_cis_by_max(feature_importances, ci_lower, ci_upper)


#plot_filtered_feature_importances(X.columns, feature_importances, ci_lower, ci_upper)

#plot_filtered_feature_importances(X.columns, feature_importances_scaled, ci_lower_scaled, ci_upper_scaled, threshold=0.3)

plot_importances(X.columns, feature_importances_scaled, feature_importances, ci_lower_scaled, ci_upper_scaled, ci_lower, ci_upper, threshold=0.1)

