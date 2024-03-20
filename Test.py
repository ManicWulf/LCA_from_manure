import dash_functions_and_callbacks as dfc
import pandas as pd



list_types = ['PW 1', 'PW 2']

list_electricity_generated = [1000, - 500]
list_heat_generated = [-5000, 10000]



label_list = ['Electricity', 'Heat']
data_df = pd.DataFrame({
    'Types': list_types,
    'Electricity': list_electricity_generated,
    'Heat': list_heat_generated,
})
title = 'Energy generation'
y_axis_title = 'Energy [kWh]'

bar_chart = dfc.create_bar_chart(label_list, data_df, title, y_axis_title)

bar_chart.show()