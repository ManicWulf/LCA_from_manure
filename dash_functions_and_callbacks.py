import plotly.express as px
import plotly.io as pio
import pandas as pd
import json
import base64
import datetime
import io
import logging
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, MATCH, Patch, ALL, dash_table, callback
from dash.dependencies import Input, Output, State

"""
Function to read uploaded .csv and .xls files and parse them into a dataframe, then displays it.
"""


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data= df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            id={"type": "new-data-upload", "index": filename},
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


# Display a Dataframe as a Table. Returns a html.Div, so it can be used as a "children" element.

def display_df_as_table(dataframe, table_name):
    return html.Div([
        html.H2(table_name),
        dash_table.DataTable(
            data=dataframe.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in dataframe.columns],
            id=table_name,
        ),
    ])


# Takes a pandas DataFrame and returns a json object for the dcc.Store component
def dataframe_to_json(dataframe):
    return dataframe.to_json(orient="records")


# Takes a json object (from dcc.Store) and creates a pandas Dataframe from it
def json_to_dataframe(json_string):
    return pd.read_json(io.StringIO(json_string), orient="records")


# Takes a csv or xls file and creates a pandas Dataframe from it

def parse_contents_to_dataframe(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return 'There was an error processing this file.'
    return df


def read_file_to_dataframe(file_path):
    """
    Read a file from a given path and return it as a pandas DataFrame.

    :param file_path: Path to the file to be read.
    :return: A pandas DataFrame containing the file data.
    """
    try:
        if file_path.endswith('.csv'):
            # Read a CSV file
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            # Read an Excel file
            df = pd.read_excel(file_path)
        else:
            return f"Unsupported file format: {file_path}"
        return df
    except Exception as e:
        print(e)
        return f"There was an error processing the file {file_path}."


# Combines the functions dataframe_to_json and parse_contents to directly create a json element from a csv or xls file, that can then be stored in a dcc.Store component

def store_contents(contents, filename):
    return dataframe_to_json(parse_contents_to_dataframe(contents, filename))


"""
Function to create a pandas dataframe
"""


def create_data_frame_calc(list_names):
    df_table = pd.DataFrame(index=list_names, columns=["name", 'value'])
    df_table["name"] = list_names
    df_table["value"] = 0.0     # initialize as floats
    return df_table


"""
function to show or hide a list of (uploaded) csv. or .xls files in a table
"""


def show_hide_buttons_for_files(show, hide, list_of_contents, list_of_names, list_of_dates, original, button_id_show,
                                button_id_hide):
    triggered_button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_button_id == button_id_show:
        if show > 0:
            if list_of_contents is not None:
                children = [
                    parse_contents(c, n, d) for c, n, d in
                    zip(list_of_contents, list_of_names, list_of_dates)]
                return children
    elif triggered_button_id == button_id_hide:
        if hide > 0:
            return []
    else:
        return original


"""
list of values needed for calculation to then create a DataFrame from it.
Then create a dataframe with the list elements as the first column.
"""

calc_values_general = ["manure_solid", "manure_liquid", "manure_straw", "methane_solid", "methane_liquid",
                       "methane_straw", "methane_tot", "methane_emissions", "nh3_emissions", "n2o_emissions", "n_tot",
                       "sum_post_storage_time", "c_tot", "c_tot_solid", "c_tot_liquid", "c_tot_straw", "methane_emissions_solid",
                       "methane_emissions_liquid", "methane_emissions_straw", "manure_tot"]

calc_values_pre_storage_transport = ["methane_solid_pre_storage", "methane_liquid_pre_storage", "methane_straw_pre_storage",
                                     "methane_tot_pre_storage", "n_tot_pre_storage", "c_tot_pre_storage",
                                     "methane_emissions_pre_storage", "nh3_emissions_pre_storage", "n2o_emissions_pre_storage"]

calc_values_post_storage_field = ["nh3_emissions_post_storage", "n2o_emissions_post_storage",
                                  "methane_emissions_post_storage", "nh3_emissions_field", "n2o_emissions_field"]

calc_values_ad = ["c_tot_ad", "methane_yield", "methane_emissions_ad", "methane_to_chp", "effective_methane_after_ad",
                  ]

calc_values_biogas_upgrading = ["biomethane_volume_tot", "biomethane_ch4", "biogas", "co2_biogas", "methane_emissions_biogas_upgrading"]

calc_values_energy = ["heat_demand_ad", "electricity_demand_ad", "electricity_demand_biogas_upgrading", "heat_demand_steam", "heat_demand_tot",
                      "electricity_demand_tot", "heat_generated_chp", "electricity_generated_chp", "electricity_generated_sofc",
                      "electricity_generated_tot", "heat_generated_tot"]

calc_values_env_impact = ["co2_methane_pre_storage", "co2_methane_post_storage", "co2_methane_field", "co2_methane_ad", "co2_methane_biogas_upgrading",
                          "co2_methane_tot", "co2_n2o_pre_storage", "co2_n2o_post_storage", "co2_n2o_ad", "co2_n2o_biogas_upgrading",
                          "co2_n2o_field", "co2_n2o_tot", "co2_electricity_mix", "co2_electricity_demand_ad", "co2_electricity_demand_biogas_upgrading",
                          "co2_electricity_demand_tot", "co2_heat_oil",
                          "co2_transport", "ubp_nh3", "ubp_co2", "ubp_electricity_demand_renew", "ubp_electricity_demand_non_renew",
                          "co2_ad_construction", "co2_chp_construction",
                          "co2_eq_tot"]

calc_values_needed = (calc_values_general + calc_values_pre_storage_transport + calc_values_post_storage_field +
                      calc_values_ad + calc_values_biogas_upgrading + calc_values_energy + calc_values_env_impact)



def create_dataframe_calc_empty():
    return create_data_frame_calc(calc_values_needed)


"""
functions to work with the config and farm files, find correct values etc.
"""

def preprocess_env_config(env_config):
    if 'name' in env_config.columns:
        # If 'name' is a column, preprocess it
        env_config['name'] = env_config['name'].str.strip().str.lower()
    return env_config



def index_env_config(env_config):
    """
    Set the 'name' column as an index for the env_config DataFrame.
    Checks if 'name' is already set as an index to avoid re-indexing.
    """
    if 'name' in env_config.columns:
        # If 'name' is a column, set it as index
        env_config.set_index('name', inplace=True)
    # If 'name' is already an index, no action is needed
    return env_config




# takes input for animal type and stable type, the name of the value (as a string) that is being searched for, and the dataframe from which to search (animal config)
# returns the value from animal config
def find_value_animal_config(animal, stable, value, animal_config):
    """Strip any whitespace from the strings and make them all lower case to prevent errors"""
    animal_config = animal_config.copy()
    animal_config['animal_type'] = animal_config['animal_type'].str.strip().str.lower()
    animal = animal.strip().lower()

    series = animal_config.query(f'animal_type == "{animal}" and stable_type == {stable}')[value]


    if series.size == 0:
        return False
    else:
        return series.iloc[0]




def find_value_animal_config_1_variable(animal,value, animal_config):

    """Strip any whitespace from the strings and make them all lower case to prevent errors"""
    animal_config = animal_config.copy()
    animal_config['animal_type'] = animal_config['animal_type'].str.strip().str.lower()
    animal = animal.strip().lower()

    series = animal_config.query(f'animal_type == "{animal}"')[value]


    if series.size == 0:
        return False
    else:
        return series.iloc[0]

# find values in env config. value is the name of the column. "value" for the actual value, "stdev" for the uncertainty
def find_value_env_config_old(factor, value, env_config):

    """Strip any whitespace from the strings and make them all lower case to prevent errors"""
    env_config['name'] = env_config['name'].str.strip().str.lower()
    factor = factor.strip().lower()

    series = env_config.query(f'name == "{factor}"')[value]
    result = series.iloc[0]

    return result


def find_value_env_config(factor, value, env_config):
    """
    Find a value in the env_config DataFrame based on the factor and value column.
    Assumes env_config is preprocessed and indexed by 'name'.
    """
    # Ensure factor is in the correct format (lowercase, stripped)
    factor = factor.strip().lower()

    # Retrieve the value using the index
    # Using .at for faster access as we are retrieving a single value
    result = env_config.at[factor, value]

    # Optional: Logging for debugging (consider reducing/removing for performance)
    #logging.debug(f'find_value_env_config: Factor="{factor}", Value="{value}", Result={result}')

    return result



# find values in farm dataframes
def find_value_farm(animal_type, value, farm):
    """Strip any whitespace from the strings and make them all lower case to prevent errors"""
    farm['name'] = farm['name'].str.strip().str.lower()
    animal_type = animal_type.strip().lower()

    series = farm.query(f'name == "{animal_type}"')[value]

    if not series.empty:
        return series.iloc[0]
    else:
        return None



# get a list of a column, without duplicate entries
# needs dataframe and column name
# returns list
def get_column_no_duplicate(df, column_name):
    column_index = df.columns.get_loc(column_name)
    column_list = df.iloc[:, column_index]
    column_list_unique = column_list.unique().tolist()
    return column_list_unique


# Get list of animal types used in animal config. For now using animal_type_ger, until data_input file is adapted to new code.
def get_animal_types(animal_config):
    animal_types = get_column_no_duplicate(animal_config, "animal_type")
    return animal_types


def add_value_to_results_df(results_df, target_row, target_column, value):
    """Add a specified value to a specific row and column in the results DataFrame."""
    results_df.loc[results_df["name"] == target_row, target_column] += value


def store_value_in_results_df(results_df, target_row, target_column, value):
    """Add a specified value to a specific row and column in the results DataFrame."""
    results_df.loc[results_df["name"] == target_row, target_column] = value


def find_value_in_results_df(results_df, target_row):
    """find a value from the specified row"""
    """Strip any whitespace from the strings and make them all lower case to prevent errors"""
    results_df['name'] = results_df['name'].str.strip().str.lower()
    target_row = target_row.strip().lower()
    return results_df.query(f'name == "{target_row}"')["value"].iloc[0]




def duplicate_store_content(source_content):
    """
    Duplicate the content of a source dcc.Store.

    Parameters:
    - source_content: The content of the source dcc.Store.

    Returns:
    - The duplicated content.
    """
    return source_content

##########################################################################
"""
Functions for plots and graphs
"""

"""Function to create a sunburst chart"""
def create_sunburst_chart(emissions, sources, values, title):
    """
    Create a sunburst chart using Plotly Express.

    Parameters:
    - emissions: List of emissions for the pie chart.
    - sources: List of sources corresponding to the emissions.
    - values: List of values for each emission-source pair.

    Returns:
    - A sunburst chart.
    """
    # Ensure the lists are of the same length
    assert len(emissions) == len(sources) == len(values), "Input lists must be of the same length"

    # Convert the lists into a DataFrame
    data = pd.DataFrame({
        'emissions': emissions,
        'sources': sources,
        'values': values
    })

    # Create the sunburst chart
    sb_chart = px.sunburst(data, path=["emissions", "sources"], values="values")

    # Set title and return the chart
    sb_chart.update_layout(title=title)
    return sb_chart


def create_bar_chart(value_label_list, data_df, title, y_axis_title):
    """
    Create a bar chart using Plotly Express.

    Parameters:
    - types: List of categories/types for the x-axis.
    - values: List of values corresponding to each type.
    - title: Title of the bar chart.
    - y_axis_title: Title of the y-axis.

    Returns:
    - A bar chart.
    """

    # Create the bar chart
    bar_chart = px.bar(data_df, x="Types", y=value_label_list, title=title)

    # Set y-axis title and return the chart
    bar_chart.update_layout(yaxis_title=y_axis_title)
    return bar_chart


def create_grouped_bar_chart(types, value_dict, title, y_axis_title):
    """
    Create a grouped bar chart using Plotly Express.

    Parameters:
    - types: List of categories/types for the x-axis.
    - value_dict: Dictionary containing value lists for each group.
                  E.g., {'Emissions': list_co2_emissions, 'Avoided burden': list_co2_avoided, ...}
    - title: Title of the bar chart.
    - y_axis_title: Title of the y-axis.

    Returns:
    - A grouped bar chart.
    """
    # Convert the lists into a DataFrame
    data = pd.DataFrame({
        'Types': types,
        **value_dict
    })

    # Create the grouped bar chart
    grouped_bar_chart = px.bar(data, x="Types", y=[col for col in data.columns if col != "Types"],
                               barmode="group", title=title)

    # Set y-axis title and return the chart
    grouped_bar_chart.update_layout(yaxis_title=y_axis_title)
    return grouped_bar_chart



