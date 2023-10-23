import pandas as pd
import os
import logging

import farm_calc as fc
import dash_functions_and_callbacks as dfc
from flask import session
import datetime
import uuid
import base64

# Ensure we have a directory for our temporary files
TEMP_DIR = "/mnt/data/temp_files/"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


# Sample callback for handling file uploads in Dash

def handle_file_upload(contents, filename):
    """
    Handle the uploaded file from dcc.Upload component.

    Parameters:
    - contents (str): Base64 encoded content of the uploaded file.
    - filename (str): Original filename of the uploaded file.
    - _session_id (str): Unique session ID for the user.

    Returns:
    - str: Path to the saved file.
    """

    # If _session_id is not provided, create a new one
    if '_session_id' not in session:
        session['_session_id'] = str(uuid.uuid4())

    _session_id = session['_session_id']

    # Create a unique filename by combining _session_id, current timestamp, and original file extension
    file_extension = filename.split('.')[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    unique_filename = f"{_session_id}_{timestamp}.{file_extension}"

    # Decode the base64 encoded content
    content_type, content_string = contents.split(',')
    decoded_content = base64.b64decode(content_string)

    # Save the decoded content to a file
    file_path = os.path.join(TEMP_DIR, unique_filename)
    with open(file_path, 'wb') as file:
        file.write(decoded_content)

    return file_path


# This function can be used in the callback for dcc.Upload to handle file uploads.
# It ensures that each uploaded file is saved with a unique filename,
# allowing multiple users to use the app simultaneously without conflicts.


def backend_calc_farms(input_filepath_list, env_config_filepath, animal_config_filepath):

    # 1. Read data from provided file paths
    input_df_list = []
    for filepath in input_filepath_list:
        input_df_list.append(dfc.read_file_to_dataframe(filepath))

    env_config_df = dfc.read_file_to_dataframe(env_config_filepath)
    animal_config_df = dfc.read_file_to_dataframe(animal_config_filepath)

    """debugging"""
    logging.debug(f'backend_calc_farm input dataframe list= {input_df_list}')
    logging.debug(f'backend_calc_farm env_config dataframe= {env_config_df}')
    logging.debug(f'backend_calc_farm animal_config dataframe = {animal_config_df}')
    """"""

    # 2. Perform the calculations using farm_calc.py functions
    results_df = fc.calc_all_farms_new(input_df_list, animal_config_df, env_config_df)
    # 3. Save results to a temporary file named after the function
    output_filepath = os.path.join(TEMP_DIR, "backend_calc_farms.csv")
    results_df.to_csv(output_filepath, index=False)

    # 4. Return the file path to the results
    return output_filepath


def backend_calc_no_treatment(input_filepath, env_config_filepath):

    # 1. Read data from provided file paths
    input_df = dfc.read_file_to_dataframe(input_filepath)
    env_config_df = dfc.read_file_to_dataframe(env_config_filepath)

    # 2. Perform the calculations using farm_calc.py functions
    results_df = fc.post_storage_and_field_emissions(input_df, env_config_df, False)
    results_df = fc.calc_env_impacts(results_df, env_config_df)

    # 3. Save results to a temporary file named after the function
    output_filepath = os.path.join(TEMP_DIR, "backend_calc_no_treatment.csv")
    results_df.to_csv(output_filepath, index=False)

    # 4. Return the file path to the results
    return output_filepath


def backend_calc_ad(input_filepath, env_config_filepath):

    # 1. Read data from provided file paths
    input_df = dfc.read_file_to_dataframe(input_filepath)
    env_config_df = dfc.read_file_to_dataframe(env_config_filepath)

    # 2. Perform the calculations using farm_calc.py functions
    results_df = fc.calc_anaerobic_digestion(input_df, env_config_df)
    results_df = fc.post_storage_and_field_emissions(results_df, env_config_df, True)
    results_df = fc.calc_chp_output(results_df, env_config_df)
    results_df = fc.calc_env_impacts(results_df, env_config_df)

    # 3. Save results to a temporary file named after the function
    output_filepath = os.path.join(TEMP_DIR, "backend_calc_ad.csv")
    results_df.to_csv(output_filepath, index=False)

    # 4. Return the file path to the results
    return output_filepath


def backend_calc_ad_biogas_upgrading(input_filepath, env_config_filepath):

    # 1. Read data from provided file paths
    input_df = dfc.read_file_to_dataframe(input_filepath)
    env_config_df = dfc.read_file_to_dataframe(env_config_filepath)

    # 2. Perform the calculations using farm_calc.py functions
    results_df = fc.calc_biogas_upgrading(input_df, env_config_df)
    results_df = fc.calc_env_impacts(results_df, env_config_df)


    # 3. Save results to a temporary file named after the function
    output_filepath = os.path.join(TEMP_DIR, "backend_calc_ad_biogas_upgrading.csv")
    results_df.to_csv(output_filepath, index=False)

    # 4. Return the file path to the results
    return output_filepath


def backend_calc_steam_treatment_ad(input_filepath, env_config_filepath):

    # 1. Read data from provided file paths
    input_df = dfc.read_file_to_dataframe(input_filepath)
    env_config_df = dfc.read_file_to_dataframe(env_config_filepath)

    # 2. Perform the calculations using farm_calc.py functions
    results_df = fc.steam_pre_treatment(input_df, env_config_df)
    results_df = fc.calc_anaerobic_digestion(results_df, env_config_df)
    results_df = fc.post_storage_and_field_emissions(results_df, env_config_df, True)
    results_df = fc.calc_chp_output(results_df, env_config_df)
    results_df = fc.calc_env_impacts(results_df, env_config_df)

    # 3. Save results to a temporary file named after the function
    output_filepath = os.path.join(TEMP_DIR, "backend_calc_steam_treatment_ad.csv")
    results_df.to_csv(output_filepath, index=False)

    # 4. Return the file path to the results
    return output_filepath


def backend_calc_steam_treatment_ad_biogas_upgrading(input_filepath, env_config_filepath):

    # 1. Read data from provided file paths
    input_df = dfc.read_file_to_dataframe(input_filepath)
    env_config_df = dfc.read_file_to_dataframe(env_config_filepath)

    # 2. Perform the calculations using farm_calc.py functions
    results_df = fc.calc_biogas_upgrading(input_df, env_config_df)
    results_df = fc.calc_env_impacts(results_df, env_config_df)

    # 3. Save results to a temporary file named after the function
    output_filepath = os.path.join(TEMP_DIR, "backend_calc_steam_treatment_ad_biogas_upgrading.csv")
    results_df.to_csv(output_filepath, index=False)

    # 4. Return the file path to the results
    return output_filepath


