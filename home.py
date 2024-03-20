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
from dash import Dash, dcc, html, Input, Output, State, MATCH, Patch, ALL, dash_table, callback, no_update
from dash.dependencies import Input, Output, State


import dash_functions_and_callbacks as dfc
from app import app
import backend_calculations as bc
from pages.data_input import data_input_layout
from pages.life_cycle_assessment import lca_layout


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',  # This will log to a file named app.log. Remove this to log to console.
                    filemode='w')  # This means the log file will be overwritten each time the app is started. Use 'a' to append.





app.server.secret_key = 'farm_life_cycle_assessment'

# Create a dictionary of page paths
page_paths = {
    "data-input-button": "/input",
    "lca-button": "/lca",
    "uncertainty-button": "/uncertainties",
    "sensitivity-analysis-button": "/sensitivity"
}
#########################################################################
"""
If the default config files are not loading, please enter the correct path in the following 2 lines of code
"""
# path to the default config files
default_animal_config_path = "default_configs/default_animal_config.xlsx"
default_environmental_config_path = "default_configs/default_environmental_config.xlsx"
###########################################################################



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
        dcc.Location(id="location", refresh=False),
        dbc.Row([
            dbc.Col(dbc.Button("Input Data", href="/input", color="primary", className="me-1"), width="auto"),
            dbc.Col(dbc.Button("Calculate LCA", href="/lca", color="primary", className="me-1"), width="auto"),
            dbc.Col(dbc.Button("Uncertainties", href="/uncertainty", color="primary", className="me-1"), width="auto"),
            dbc.Col(dbc.Button("Sensitivity Analysis", href="/sensitivity-analysis", color="primary", className="me-1"),
                    width="auto"),
        ]),
        html.Br(),
        html.Br(),
        # Dynamic content container
        dash.page_container,
        html.Div(id="output-config-upload"),
        dcc.Store(id='config-paths', data={"environmental_config": default_environmental_config_path,
                                           "animal_config": default_animal_config_path}),
    ])


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


    # Start the main app
    app.run_server(debug=True)


