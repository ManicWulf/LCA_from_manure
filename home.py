"""
Main file, coordinates and communicates between the other files. Creates the first page of the dash app.
Used to upload the config files and navigate between the different applications,
namely Data-Input, LCA generation, Uncertainty (with monte-carlo), and Sensitivity analysis
"""

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


import dash_functions_and_callbacks as dfc
import backend_calculations as bc


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',  # This will log to a file named app.log. Remove this to log to console.
                    filemode='w')  # This means the log file will be overwritten each time the app is started. Use 'a' to append.

# Style sheet used: , external_stylesheets=[dbc.themes.MATERIA]
# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

app.server.secret_key = 'farm_life_cycle_assessment'

# Create a dictionary of page paths
page_paths = {
    "data-input-button": "/input",
    "lca-button": "/lca",
    "uncertainty-button": "/uncertainties",
    "sensitivity-analysis-button": "/sensitivity"
}

# path to the default config files
default_animal_config_path = "default_configs/default_animal_config.xlsx"
default_environmental_config_path = "default_configs/default_environmental_config.xlsx"

# Read the default configs
default_animal_config = pd.read_excel(default_animal_config_path).to_json(date_format='iso', orient='records')
default_environmental_config = pd.read_excel(default_environmental_config_path).to_json(date_format='iso', orient='records')



# Start the main app
if __name__ == "__main__":
    app.layout = html.Div([
        html.H1("Life Cycle Assessment of biogas plants using animal manure"),
        dcc.Upload(
                id='upload-config',
                children=html.Div([
                    'Please upload your animal config file and your environmental config file. Drag and Drop or ',
                    html.A('Click to Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
        dcc.Location(id="location"),
        html.Button('Input Data', id='data-input-button',  n_clicks=0),
        html.Button('Calculate LCA', id='lca-button', n_clicks=0),
        html.Button('Uncertainties', id='uncertainty-button', n_clicks=0),
        html.Button('Sensitivity Analysis', id='sensitivity-analysis-button', n_clicks=0),
        html.Br(),
        html.Br(),
        dash.page_container,
        html.Div(id="output-config-upload"),
        dcc.Store(id='config-paths', data={"environmental_config": default_environmental_config_path,
                                                 "animal_config": default_animal_config_path}),
        dcc.Store(id='animal-config-store', data=default_animal_config),
        dcc.Store(id='environmental-config-store', data=default_environmental_config),
    ])


    """
    callback to navigate the pages with the buttons defined earlier
    """
    @app.callback(
        Output("location", "href"),
        Input("data-input-button", "n_clicks"),
        Input("lca-button", "n_clicks"),
        Input("uncertainty-button", "n_clicks"),
        Input("sensitivity-analysis-button", "n_clicks"),
        State("location", "href")
    )
    def navigate_to_page(
            data_input_button_n_clicks,
            lca_button_n_clicks,
            uncertainty_button_n_clicks,
            sensitivity_analysis_button_n_clicks,
            href
    ):
        #Navigates to the specified page.

        # Get the ID of the button that triggered the callback
        triggered_button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]

        # Check if the triggered button ID is empty
        if not triggered_button_id:
            return href

        # Get the page path for the triggered button
        page_path = page_paths[triggered_button_id]

        # Return the page path
        return page_path



    """
    Store the uploaded data in their respective dcc.Store components
    """


    @callback(
        Output('config-paths', 'data'),  # Output component to store the file paths
        Input('upload-config', 'contents'),  # Input component for file upload
        State('upload-config', 'filename')  # State to get the filename of the uploaded file
    )
    def update_config_upload_files(contents_list, filename_list):
        if contents_list:
            saved_file_paths_dict = {"environmental_config": default_environmental_config_path,
                                     "animal_config": default_animal_config_path}
            for contents, filename in zip(contents_list, filename_list):
                # Call the handle_file_upload function to save the uploaded farm file
                if 'env' in filename:
                    saved_file_paths_dict['environmental_config'] = bc.handle_file_upload(contents, filename)
                elif 'animal' in filename:
                    saved_file_paths_dict['animal_config'] = bc.handle_file_upload(contents, filename),

            # Return a message to the user indicating where the file was saved
            return saved_file_paths_dict
        return dash.no_update  # if no data, return without updating



    # Connect the dcc.Upload component's contents output to the input of two callback functions
    @app.callback(
        Output('animal-config-store', 'data'),
        Input('upload-config', 'contents'),
        State('upload-config', 'filename'),
        State('animal-config-store', 'data'),
        prevent_initial_callbacks=True
    )
    def store_animal_config_data(list_of_contents, list_of_names, initial_data):
        # Check the file name of the uploaded file
        if list_of_contents is not None:
            for contents, file_name in zip(list_of_contents, list_of_names):

                # If the file name is 'animal_config.xlsx', parse the file content to a dataframe, then store it in config store
                if file_name == 'animal_config.xlsx':
                    return dfc.store_contents(contents, file_name)

        # Otherwise, return initial data (default config)
        return initial_data


    @app.callback(
        Output('environmental-config-store', 'data'),
        Input('upload-config', 'contents'),
        State('upload-config', 'filename'),
        State('environmental-config-store', 'data'),
        prevent_initial_callbacks=True
    )
    def store_environmental_config_data(list_of_contents, list_of_names, initial_data):
        # Check the file name of the uploaded file
        if list_of_contents is not None:
            for contents, file_name in zip(list_of_contents, list_of_names):

                # If the file name is 'environmental_config.xlsx', parse the file content to a dataframe, then store it in config store
                if file_name == 'environmental_config.xlsx':
                    return dfc.store_contents(contents, file_name)

        # Otherwise, return initial data (default config)
        return initial_data




    # Start the main app
    app.run_server(debug=True)


