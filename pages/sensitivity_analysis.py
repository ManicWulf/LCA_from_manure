import dash
from dash import html



dash.register_page(__name__, path="/sensitivity")

layout = html.Div([
    html.H1('This is our sensitivity analysis page'),
    html.Div('This is our sensitivity analysis content.'),
])
