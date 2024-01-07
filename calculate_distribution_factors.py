import pandas as pd
import numpy as np

def calculate_mu_sigma_with_median(row):
    if row['Distribution function'] == 'lognormal' and not np.isnan(row['median']):
        median = row['median']

        # Case 1: Using upper and lower boundaries
        if not np.isnan(row['upper']) and not np.isnan(row['lower']):
            sigma = np.sqrt(np.log(row['upper'] / row['lower'])) / 2

        # Case 2: Using stdev95
        elif not np.isnan(row['stdev95']):
            stdev95 = row['stdev95']
            sigma = np.sqrt(np.log(1 + (stdev95/median)**2))

        # Case 3: Using n and se
        elif not np.isnan(row['n']) and not np.isnan(row['se']):
            sigma = row['se'] * np.sqrt(row['n']) / median

        else:
            # No sufficient data for calculation
            return pd.Series([np.nan, np.nan])

        mu = np.log(median) - sigma**2 / 2
        return pd.Series([mu, sigma])

    else:
        return pd.Series([np.nan, np.nan])

def main():
    # File paths
    original_file_path = 'default_configs/combined_config.xlsx'  # Replace with your file path
    updated_file_path = 'default_configs/updated_combined_config.xlsx'  # Replace with your desired output path

    # Load the Excel file
    df = pd.read_excel(original_file_path)

    # Apply the function to each row and update 'mu' and 'sigma' columns
    df[['mu', 'sigma']] = df.apply(calculate_mu_sigma_with_median, axis=1)

    # Save the updated DataFrame to a new Excel file
    df.to_excel(updated_file_path, index=False)

    print(f"Updated file saved to: {updated_file_path}")

if __name__ == "__main__":
    main()
