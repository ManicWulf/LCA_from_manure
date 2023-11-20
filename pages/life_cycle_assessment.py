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
    html.Div(id='output_calc_files'),
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
    Input('stored-farm-file-paths', 'data'),
    State('config-paths', 'data'),
    config_prevent_initial_callbacks=True
)
def lca_calculations(farm_paths_list, config_paths_dict):
    """

    :param farm_paths_list:
    :param config_paths_dict:
    :return:
    """
    if farm_paths_list is not None:
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


@callback(
    Output('graph-container', 'children'),
    Input('calculate-lca-button', 'n_clicks'),
    State('calc-data-file-paths', 'data'),
    prevent_initial_callback=True
)
def plot_co2_sunburst(n_clicks, calc_file_paths_dict):
    """
    generate the sunburst charts for CO2-eq. emissions
    :param calc_file_paths_dict: dictionary with the farm file paths
    :return:
    """
    if n_clicks > 0:

        # create list with titles for the charts
        title_list = ['CO2 Sunburst chart for No treatment pathway', 'CO2 Sunburst chart for ad only pathway',
                      'CO2 Sunburst chart for AD with biogas upgrading pathway', 'CO2 Sunburst chart for steam pretreatment with AD pathway',
                      'CO2 Sunburst chart for steam pretreatment with AD and Biogas upgrading pathway']

        # remove the first entry of the calc_file_dict, since we won't be needing the 'farm-calc' entry
        calc_file_paths_dict.pop('farm-calc', None)

        # create empty list for returning the results
        graphs_list = []
        dataframe_list = []

        # iterate through the calc_file dict and title list to create the charts
        for (key, filepath), title in zip(calc_file_paths_dict.items(), title_list):
            if filepath is not None:
                dataframe = dfc.read_file_to_dataframe(filepath)
                dataframe_list.append(dataframe)
                graphs_list.append(dcc.Graph(id=title, figure=pac.sunburst_co2(dataframe, title)))

        graphs_list.append(pac.create_bar_chart_list(dataframe_list))
        return graphs_list


    return dash.no_update


@callback(
    Output('output_calc_files', 'children'),
    Input('calculate-lca-button', 'n_clicks'),
    State('calc-data-file-paths', 'data'),
)
def show_calc_files_debug(n_clicks, calc_file_paths_dict):
    if n_clicks > 0:
        children = []
        for key, filepath in calc_file_paths_dict.items():
            if filepath is not None:

                calc_file_df = dfc. read_file_to_dataframe(filepath)
                children.append(html.Div([dfc.display_df_as_table(calc_file_df, key)]))

        return children
    return dash.no_update

""" 
Display a table with the uploaded farm files, when clicking on the show button, Hide when clicking on Hide button
"""


@callback(
    Output('output-data-upload-lca', 'children'),
    Input("show-farm-files-button", "n_clicks"),
    Input("hide-farm-files-button", "n_clicks"),
    State('stored-farm-file-paths', 'data'),
    prevent_initial_callbacks=True
)
def update_output(show, hide, list_of_farm_paths):
    triggered_button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_button_id == "show-farm-files-button":
        if show > 0:
            if list_of_farm_paths is not None:
                children = []
                for path, i in zip(list_of_farm_paths, range(len(list_of_farm_paths))):
                    farm = dfc.read_file_to_dataframe(path)
                    children.append(html.Div([dfc.display_df_as_table(farm, f"farm{i}")]))
                return children
    elif triggered_button_id == "hide-farm-files-button":
        if hide > 0:
            return []
    else:
        return

"""
Display the config file, when you click on the Show button, and hide them when you click on the hide button.
"""

@callback(
    Output("output-data-home", "children"),
    Input("output-data-home-button", "n_clicks"),
    Input("hide-data-home-button", "n_clicks"),
    State('config-paths', "data"),
    prevent_initial_callbacks=True
)
def output_config_home(show, hide, config_paths):
    triggered_button_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_button_id == "output-data-home-button":
        if show > 0:
            env_data = dfc.read_file_to_dataframe(config_paths['environmental_config'])
            animal_data = dfc.read_file_to_dataframe(config_paths['animal_config'])
            children = html.Div([dfc.display_df_as_table(env_data, "Environmental Config"),
                                 dfc.display_df_as_table(animal_data, "Animal Config")])
            return children

    elif triggered_button_id == "hide-data-home-button":
        if hide > 0:
            return []
    else:
        return


