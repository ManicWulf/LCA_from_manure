import plotly.express as px
import plotly.io as pio
import pandas as pd
import json
import base64
import datetime
import io
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, MATCH, Patch, ALL, dash_table, callback
from dash.dependencies import Input, Output, State



dash.register_page(__name__, path="/input")
"""
animal_classes = ["Milchkuh", "Mutterkuh", "Aufzuchtrind", "Mastkalb", "Mutterkuhkalb", "RiendviehMast", "Zuchtstier", "Mastschwein", "Zuchtschweineplatz", "Legehenne", "Junghenne", "Mastpoulet"]
cattle = ["Milchkuh", "Mutterkuh", "Aufzuchtrind", "Mastkalb", "Mutterkuhkalb", "RiendviehMast", "Zuchtstier"]
pigs = ["Mastschwein", "Zuchtschweineplatz"]
poultry = ["Legehenne", "Junghenne", "Mastpoulet"]
"""

cattle = ["milk_cow", "mother_cow", "heifer", "fattening_calf", "suckler_calf", "beef_cow", "breeding_bull"]
pigs = ["fattening_pig", "breeding_pig"]
poultry = ["laying_hen", "pullet", "broiler_chicken"]

animal_classes = cattle + pigs + poultry

dataframe_columns = ["name", "num-animals", "days-outside", "hours-outside", "manure-type"]





def create_data_frame(animal):
    df_table = pd.DataFrame(columns=dataframe_columns)
    df_table["name"] = animal
    df_table = df_table.fillna(0)
    return df_table


def create_accordion(n):
    accordion = []
    for i in range(n):
        new_item = create_accordion_item(i)
        accordion += new_item
    return accordion


def create_accordion_item(i):
    data_table = []
    data_table += generate_data_table(i, "cattle", cattle)
    data_table += generate_data_table(i, "pigs", pigs)
    data_table += generate_data_table(i, "poultry", poultry)

    item = [dbc.AccordionItem(
        children=[dbc.Container([
            dbc.Row([
                dbc.Col([
                    "How many days is the manure stored before processing? Please provide a minimum, maximum and expected value.",
                    dcc.Input(id={"type": 'days-pre-stored-minimum', "index": i+1}, type='number', placeholder='Minimum days'),
                    dcc.Input(id={"type": 'days-pre-stored-expected', "index": i + 1}, type='number', placeholder='Expected days'),
                    dcc.Input(id={"type": 'days-pre-stored-maximum', "index": i + 1}, type='number', placeholder='Maximum days')
                ]),
                dbc.Col([
                    "How many days is the manure or digestate stored after processing and before field application? Please provide a minimum, maximum and expected value.",
                    dcc.Input(id={"type": 'days-post-stored-minimum', "index": i + 1}, type='number', placeholder='Minimum days'),
                    dcc.Input(id={"type": 'days-post-stored-expected', "index": i + 1}, type='number', placeholder='Expected days'),
                    dcc.Input(id={"type": 'days-post-stored-maximum', "index": i + 1}, type='number', placeholder='Maximum days')
                ]),
                dbc.Col([
                    html.Button(f"Store data for farm {i+1}", id={"type": "store-data-button", "index": i+1}, n_clicks=0)
                ])
            ]),
            dcc.Download(id={"type": "download-data-farm", "index": i+1}),
            dbc.Row([
                dbc.Col([
                    "How far is the farm from the processing location in km? Please provide a minimum, maximum and expected value.",
                    dcc.Input(id={"type": 'distance-minimum', "index": i + 1}, type='number', placeholder= 'Minimum distance'),
                    dcc.Input(id={"type": 'distance-expected', "index": i + 1}, type='number', placeholder= 'Expected distance'),
                    dcc.Input(id={"type": 'distance-maximum', "index": i + 1}, type='number', placeholder= 'Maximum distance')
                ]),
                dbc.Col([
                    "Select the manure application method:",
                    dcc.Dropdown(
                        id={"type": 'manure-application-method', "index": i + 1},
                        options=[
                            {'label': 'Splash Plate', 'value': 'splash_plate'},
                            {'label': 'Drag Hose', 'value': 'drag_hose'},
                            {'label': 'Drag Shoe', 'value': 'drag_shoe'},
                        ],
                        value='drag_hose'  # Optional: default value
                    )
                ]),
                dbc.Col([
                    "Select the manure transportation method:",
                    dcc.Dropdown(
                        id={"type": 'manure-transport-method', "index": i + 1},
                        options=[
                            {'label': 'Tractor with 10t lorry', 'value': 'tractor'},
                            {'label': 'Truck', 'value': 'truck'},
                            {'label': 'Other alternative', 'value': 'other'},
                        ],
                        value='tractor'  # Optional: default value
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    "What is the expected heat demand of potential costumers surrounding the AD plant in kWh? Please provide a value for summer and winter.",
                    dcc.Input(id={"type": 'heat-demand-summer', "index": i + 1}, type='number',
                              placeholder='Summer (kWh)'),
                    dcc.Input(id={"type": 'heat-demand-winter', "index": i + 1}, type='number',
                              placeholder='Winter (kWh)'),
                    ])
            ]),
            dbc.Row(
                data_table
            )
        ])],
        title=f"Farm-{i+1}",
        item_id=f"farm-accordion-item-{i+1}",
        id={"type": "farm-accordion-item", "index": i+1}
    )]
    return item


def generate_data_table(k, animals, animals_list):
    data = create_data_frame(animals_list)
    data_table = [html.Br(), html.H3(f"{animals}"), dash_table.DataTable(

        data=data.to_dict("records"),
        columns=[
            {"name": f"Type of {animals}", "id": "name", "editable": False},
            {"name": f"Number of animals", "id": "num-animals", "type": "numeric"},
            {"name": "Avg days outside stable per year", "id": "days-outside", "type": "numeric"},
            {"name": "Avg hours outside per day", "id": "hours-outside", "type": "numeric"},
            {"name": "Type of manure collected", "id": "manure-type", "presentation": "dropdown"},
        ],
        editable=True,
        dropdown={
            "manure-type": {
                "options": [
                    {"label": "Only solid", "value": 2},
                    {"label": "Only liquid", "value": 0},
                    {"label": "Mixed", "value": 1}
                ],
            }
        },
        id={"type": f"data-table-{animals}", "index": k+1}
    )]
    return data_table


def datatable_to_dataframe(dt_1, dt_2, dt_3, pre_storage_min, pre_storage_expected, pre_storage_max, post_storage_min,
                            post_storage_expected, post_storage_max, distance_min, distance_expected, distance_max,
                           application_method, transport_method, heat_demand_winter, heat_demand_summer, country):
    df_1 = pd.DataFrame.from_records(dt_1)
    df_2 = pd.DataFrame.from_records(dt_2)
    df_3 = pd.DataFrame.from_records(dt_3)
    df = pd.concat([df_1, df_2, df_3])
    df = add_manure_storage_cell(df, pre_storage_min, pre_storage_expected, pre_storage_max, post_storage_min,
                            post_storage_expected, post_storage_max, distance_min, distance_expected, distance_max,
                                 application_method, transport_method, heat_demand_winter, heat_demand_summer, country)
    return df


def add_manure_storage_cell(df, pre_storage_min, pre_storage_expected, pre_storage_max, post_storage_min,
                            post_storage_expected, post_storage_max, distance_min, distance_expected, distance_max,
                            application_method, transport_method, heat_demand_winter, heat_demand_summer, country):
    df["additional-data"] = 0
    df.loc["pre_min"] = 0
    df.at["pre_min", "additional-data"] = pre_storage_min
    df.at["pre_min", "name"] = "pre_storage_min"
    df.loc["pre_expected"] = 0
    df.at["pre_expected", "additional-data"] = pre_storage_expected
    df.at["pre_expected", "name"] = "pre_storage_expected"
    df.loc["pre_max"] = 0
    df.at["pre_max", "additional-data"] = pre_storage_max
    df.at["pre_max", "name"] = "pre_storage_max"
    df.loc["post_min"] = 0
    df.at["post_min", "additional-data"] = post_storage_min
    df.at["post_min", "name"] = "post_storage_min"
    df.loc["post_expected"] = 0
    df.at["post_expected", "additional-data"] = post_storage_expected
    df.at["post_expected", "name"] = "post_storage_expected"
    df.loc["post_max"] = 0
    df.at["post_max", "additional-data"] = post_storage_max
    df.at["post_max", "name"] = "post_storage_max"
    df.loc["dist_min"] = 0
    df.at["dist_min", "additional-data"] = distance_min
    df.at["dist_min", "name"] = "distance_min"
    df.loc["dist_expected"] = 0
    df.at["dist_expected", "additional-data"] = distance_expected
    df.at["dist_expected", "name"] = "distance_expected"
    df.loc["dist_max"] = 0
    df.at["dist_max", "additional-data"] = distance_max
    df.at["dist_max", "name"] = "distance_max"
    df.loc["application_method"] = 0
    df.at["application_method", "additional-data"] = application_method
    df.at["application_method", "name"] = "application_method"
    df.loc["transport_method"] = 0
    df.at["transport_method", "additional-data"] = transport_method
    df.at["transport_method", "name"] = "transport_method"
    df.loc["heat_demand_winter"] = 0
    df.at["heat_demand_winter", "additional-data"] = heat_demand_winter
    df.at["heat_demand_winter", "name"] = "heat_demand_winter"
    df.loc["heat_demand_summer"] = 0
    df.at["heat_demand_summer", "additional-data"] = heat_demand_summer
    df.at["heat_demand_summer", "name"] = "heat_demand_summer"
    df.loc["country"] = 0
    df.at["country", "additional-data"] = country
    df.at["country", "name"] = "country"
    return df


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
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            id={"type": "farm-data-upload", "index": filename},
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])



def data_input_layout():
    layout = html.Div([
        html.H3([
            "Enter the number of farms.",
            dcc.Input(id='num-farms', type='number', value=1),
            html.Button('Submit', id='submit-button', n_clicks=0),
            html.Br(),
        ]),
        dbc.Row([
            dbc.Col([
                "Select your country:",
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[
                        {'label': 'Switzerland', 'value': 'ch'},
                        {'label': 'Germany', 'value': 'ge'},
                        {'label': 'Austria', 'value': 'at'},
                    ],
                    value='ch'  # Optional: default value
                )
            ]),
        ]),
        dcc.Upload(
            id='upload-data',
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
        #dcc.Tabs(id='farm-tabs'),  #create the tabs for the different farms
        dbc.Accordion(id="farm-accordion"),
        #html.Div(id='output-data-upload', style={'display': 'none'})
        html.Div(id='output-data-upload'),
        html.Div(id="output-farm-data-calc")
    ])
    return layout


layout = data_input_layout()

@callback(
    Output("farm-accordion", "children"),
    Input('submit-button', 'n_clicks'),
    State('num-farms', 'value'),
)
def generate_accordion(n_clicks, num_farms):
    if n_clicks:
        accordion_items = create_accordion(num_farms)
        return accordion_items


# Download data into csv file
@callback(
    Output({"type": "download-data-farm", "index": MATCH}, "data"),
    Input({"type": f'store-data-button', "index": MATCH}, 'n_clicks'),
    State({"type": 'data-table-cattle', "index": MATCH}, 'data'),
    State({"type": 'data-table-pigs', "index": MATCH}, 'data'),
    State({"type": 'data-table-poultry', "index": MATCH}, 'data'),
    State({"type": 'days-pre-stored-minimum', "index": MATCH}, "value"),
    State({"type": 'days-pre-stored-expected', "index": MATCH}, "value"),
    State({"type": 'days-pre-stored-maximum', "index": MATCH}, "value"),
    State({"type": 'days-post-stored-minimum', "index": MATCH}, "value"),
    State({"type": 'days-post-stored-expected', "index": MATCH}, "value"),
    State({"type": 'days-post-stored-maximum', "index": MATCH}, "value"),
    State({"type": 'distance-minimum', "index": MATCH}, "value"),
    State({"type": 'distance-expected', "index": MATCH}, "value"),
    State({"type": 'distance-maximum', "index": MATCH}, "value"),
    State({"type": "manure-application-method", "index": MATCH}, "value"),
    State({"type": "manure-transport-method", "index": MATCH}, "value"),
    State({"type": "heat-demand-winter", "index": MATCH}, "value"),
    State({"type": "heat-demand-summer", "index": MATCH}, "value"),
    State("country-dropdown", "value"),
    State({"type": "farm-accordion-item", "index": MATCH}, "title"),
    prevent_initial_call=True
)
def download_data_table_csv(n_clicks, dt_cattle, dt_pigs, dt_poultry, pre_storage_min, pre_storage_expected, pre_storage_max, post_storage_min,
                            post_storage_expected, post_storage_max, distance_min, distance_expected, distance_max, application_method, transport_method, heat_demand_winter, heat_demand_summer, country, title):
    if n_clicks:
        df = datatable_to_dataframe(dt_cattle, dt_pigs, dt_poultry, pre_storage_min, pre_storage_expected, pre_storage_max, post_storage_min,
                            post_storage_expected, post_storage_max, distance_min, distance_expected, distance_max, application_method, transport_method, heat_demand_winter, heat_demand_summer, country)
        return dcc.send_data_frame(df.to_csv, f"{title}.csv")


""" 
Display a table with the uploaded files
"""
@callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


