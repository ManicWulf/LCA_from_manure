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
import farm_calc
import plots_and_charts as pac
import backend_calculations as bc





dash.register_page(__name__, path="/lca")

layout = html.Div([
    html.H1('Life cycle assessment'),
    html.Div('Upload your farm.csv files here.'),
    dcc.Upload(
        id='upload-data-lca',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
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
    html.Button("Show Config files", id="output-data-home-button", n_clicks=0),
    html.Button("Hide Config files", id="hide-data-home-button", n_clicks=0),
    html.Br(),
    html.Br(),
    html.Button("Show farm files", id="show-farm-files-button", n_clicks=0),
    html.Button("Hide farm files", id="hide-farm-files-button", n_clicks=0),
    html.Br(),
    html.Br(),
    html.Button("Calculate LCA", id="calculate-lca-button", n_clicks=0),
    html.Br(),
    html.Div(id='output-data-upload-lca'),
    html.Br(),
    html.Div(id="output-data-home"),
    html.Div(id="file-upload-output"),

    # define all the store elements to store data frames
    dcc.Store(id="stored-farm-file-paths"),
    dcc.Store(id="calc-data-file-paths", data={'farm-calc': None,
                                               'no-treatment-calc': None,
                                               'ad-only-calc': None,
                                               'ad-biogas-upgrading-calc': None,
                                               'steam-ad-calc': None,
                                               'steam-ad-biogas-upgrading-calc': None}),

    # define the outputs
    html.Div(id="graph-container"),
    html.Div(id="test-output")
])

""" 
Display a table with the uploaded farm files, when clicking on the show button, Hide when clicking on Hide button
"""


@callback(
    [Output('file-upload-output', 'children'),  # Output component to display a message to the user
     Output('stored-farm-file-paths', 'data')],     # Output component to store the file paths
    Input('upload-data-lca', 'contents'),  # Input component for file upload
    State('upload-data-lca', 'filename')  # State to get the filename of the uploaded file
)
def update_farm_upload_files(contents_list, filename_list):
    if contents_list:
        saved_file_paths = []
        for contents, filename in zip(contents_list, filename_list):
            # Call the handle_file_upload function to save the uploaded farm file
            saved_file_paths.append(bc.handle_file_upload(contents, filename))

        # Return a message to the user indicating where the file was saved
        return f"Uploaded farm files saved to: {saved_file_paths}", saved_file_paths
    return "Please upload a valid farm file.", None


@callback(
    Output('calc-data-file-paths', 'data'),
    Input("calculate-lca-button", "n_clicks"),
    State('stored-farm-file-paths', 'data'),
    State('config-paths', 'data')
)
def lca_calculations(n_clicks, farm_paths_list, config_paths_dict):
    """

    :param farm_paths_list:
    :param config_paths_dict:
    :return:
    """
    if n_clicks > 0:
        # read file paths for config files
        env_config_path = config_paths_dict['environmental_config']
        animal_config_path = config_paths_dict['animal_config']

        # create empty calculation file paths dictionary
        calc_file_paths_dict = {'farm-calc': bc.backend_calc_farms(farm_paths_list, env_config_path, animal_config_path),
                                'no-treatment-calc': None,
                                'ad-only-calc': None,
                                'ad-biogas-upgrading-calc': None,
                                'steam-ad-calc': None,
                                'steam-ad-biogas-upgrading-calc': None}
        calc_file_paths_dict['no-treatment-calc'] = bc.backend_calc_no_treatment(calc_file_paths_dict['farm-calc'], env_config_path)
        calc_file_paths_dict['ad-only-calc'] = bc.backend_calc_ad(calc_file_paths_dict['farm-calc'], env_config_path)
        calc_file_paths_dict['ad-biogas-upgrading-calc'] = bc.backend_calc_ad_biogas_upgrading(calc_file_paths_dict['ad-only-calc'], env_config_path)
        calc_file_paths_dict['steam-ad-calc'] = bc.backend_calc_steam_treatment_ad(calc_file_paths_dict['farm-calc'], env_config_path)
        calc_file_paths_dict['steam-ad-biogas-upgrading-calc'] = bc.backend_calc_steam_treatment_ad_biogas_upgrading(calc_file_paths_dict['steam-ad-calc'], env_config_path)

        return calc_file_paths_dict

    return dash.no_update



