"""
File for all plots and charts

"""

import pandas as pd
from dash import dcc

import dash_functions_and_callbacks as dfc


def sunburst_co2_old(untreated, anaerobic_digestion, ad_biogas, steam_ad, steam_ad_biogas):
    """
    :param untreated: df of calculations for untreated/business as usual
    :param anaerobic_digestion: df of calcs for AD only
    :param ad_biogas: df of calcs for ad with biogas upgrading
    :param steam_ad: df of calcs for steam pretreatment with AD
    :param steam_ad_biogas: df of cals for steam pretreatment with AD and biogas upgrading
    :return: Sunburst charts of CO2 emissions for all pathways
    """
    """start by creating list of the labels"""
    labels_emissions_sb_untreated = ["N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O", "CH\u2084", "CH\u2084", "CH\u2084"]
    labels_sources_sb_untreated = ["Pre storage", "AD process", "Post storage", "Field application", "Pre storage",
                                   "Post storage", "Field application"]

    labels_emissions_sb_ad = ["N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O", "CH\u2084", "CH\u2084", "CH\u2084",
                              "CH\u2084", "Electricity", "Transport"]
    labels_sources_sb_ad = ["Pre storage", "AD process", "Post storage", "Field application", "Pre storage",
                            "AD process", "Post storage", "Field application", "Anaerobic Digestion", None]

    labels_emissions_sb_upgrading = ["N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O",
                                     "CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084",
                                     "Electricity", "Electricity",
                                     "Transport"]
    labels_sources_sb_upgrading = ["Pre storage", "AD process", "Post storage", "Field application",
                                   "Pre storage", "AD process", "Biogas upgrading", "Post storage", "Field application",
                                   "Anaerobic Digestion", "Biogas upgrading",
                                   None]

    labels_emissions_sb_steam = ["N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O",
                                 "CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084",
                                 "Electricity",
                                 "Transport"]
    labels_sources_sb_steam = ["Pre storage", "AD process", "Post storage", "Field application",
                               "Pre storage", "AD process", "Post storage", "Field application",
                               "Anaerobic Digestion",
                               None]

    labels_emissions_sb_upgrading_steam = ["N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O",
                                           "CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084",
                                           "Electricity", "Electricity",
                                           "Transport"]
    labels_sources_sb_upgrading_steam = ["Pre storage", "AD process", "Post storage", "Field application",
                                         "Pre storage", "AD process", "Biogas upgrading", "Post storage",
                                         "Field application",
                                         "Anaerobic Digestion", "Biogas upgrading",
                                         None]

    """generate SB chart untreated"""


    #n2o sources
    co2_n2o_pre_storage = dfc.find_value_in_results_df(untreated, "co2_n2o_pre_storage")
    return


def sunburst_co2(input_df, title):
    labels_emissions = ["CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084", "CH\u2084",
                        "N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O", "N\u2082O",
                        "Electricity", "Electricity", "Transport", 'Construction', 'Construction'
                        ]
    labels_sources = ["Pre storage", "AD process", "Biogas upgrading", "Post storage", "Field application",
                      "Pre storage", "AD process", "Biogas upgrading", "Post storage", "Field application",
                      "Anaerobic Digestion", "Biogas upgrading",
                      None,
                      'AD plant', 'CHP generator'
                      ]
    """get relevant values from input df"""

    names_value_list = ["co2_methane_pre_storage", "co2_methane_ad", "co2_methane_biogas_upgrading",
                        "co2_methane_post_storage", "co2_methane_field",
                        "co2_n2o_pre_storage", "co2_n2o_ad", "co2_n2o_biogas_upgrading", "co2_n2o_post_storage",
                        "co2_n2o_field",
                        "co2_electricity_demand_ad", "co2_electricity_demand_biogas_upgrading",
                        "co2_transport",
                        'co2_ad_construction', 'co2_chp_construction']

    """create values list"""
    values = []
    for value in names_value_list:
        values.append(round(dfc.find_value_in_results_df(input_df, value), 2))

    return dfc.create_sunburst_chart(labels_emissions, labels_sources, values, title)


def bar_chart_n2o(list_input_df):
    """
    Creates a bar chart of N2O emissions by the different pathways
    :param untreated_df:
    :param ad_df:
    :param steam_df:
    :param biogas_df:
    :param steam_biogas_df:
    :return: bar chart of N2O emissions
    """
    list_types = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

    list_n2o_pre_storage = []
    list_n2o_post_storage = []
    list_n2o_field = []

    for df in list_input_df:
        list_n2o_pre_storage.append(dfc.find_value_in_results_df(df, 'co2_n2o_pre_storage'))
        list_n2o_post_storage.append(dfc.find_value_in_results_df(df, 'co2_n2o_post_storage'))
        list_n2o_field.append(dfc.find_value_in_results_df(df, 'co2_n2o_field'))

    label_list = ['Pre Storage', 'Post Storage', 'Field application']
    data_df = pd.DataFrame({
        'Types': list_types,
        'Pre Storage': list_n2o_pre_storage,
        'Post Storage': list_n2o_post_storage,
        'Field application': list_n2o_field
    })
    title = 'N\u2082O emissions'
    y_axis_title = 'N\u2082O emissions [kg N\u2082O]'

    return dfc.create_bar_chart(label_list, data_df, title, y_axis_title)


def bar_chart_nh3(list_input_df):
    """
    Creates a bar chart of NH3 emissions by the different pathways
    :param untreated_df:
    :param ad_df:
    :param steam_df:
    :param biogas_df:
    :param steam_biogas_df:
    :return: bar chart of N2O emissions
    """
    list_types = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

    list_nh3_pre_storage = []
    list_nh3_post_storage = []
    list_nh3_field = []

    for df in list_input_df:
        list_nh3_pre_storage.append(dfc.find_value_in_results_df(df, 'nh3_emissions_pre_storage'))
        list_nh3_post_storage.append(dfc.find_value_in_results_df(df, 'nh3_emissions_post_storage'))
        list_nh3_field.append(dfc.find_value_in_results_df(df, 'nh3_emissions_field'))

    label_list = ['Pre Storage', 'Post Storage', 'Field application']
    data_df = pd.DataFrame({
        'Types': list_types,
        'Pre Storage': list_nh3_pre_storage,
        'Post Storage': list_nh3_post_storage,
        'Field application': list_nh3_field
    })
    title = 'NH\u2083 emissions'
    y_axis_title = 'NH\u2083 emissions [kg NH\u2083]'

    return dfc.create_bar_chart(label_list, data_df, title, y_axis_title)


def bar_chart_co2(list_input_df):
    """
    Creates a bar chart of co2 eq emissions by the different pathways
    :param untreated_df:
    :param ad_df:
    :param steam_df:
    :param biogas_df:
    :param steam_biogas_df:
    :return: bar chart of N2O emissions
    """
    list_types = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

    list_co2_n2o = []
    list_co2_ch4 = []
    list_co2_transport = []
    list_co2_electricity = []
    list_co2_construction_ad = []
    list_co2_construction_chp = []
    list_co2_electricity_substituted = []
    list_co2_heat_substituted = []

    for df in list_input_df:
        list_co2_n2o.append(dfc.find_value_in_results_df(df, 'co2_n2o_tot'))
        list_co2_ch4.append(dfc.find_value_in_results_df(df, 'co2_methane_tot'))
        list_co2_transport.append(dfc.find_value_in_results_df(df, 'co2_transport'))
        list_co2_electricity.append(dfc.find_value_in_results_df(df, 'co2_electricity_demand_tot'))
        list_co2_construction_ad.append(dfc.find_value_in_results_df(df, 'co2_ad_construction'))
        list_co2_construction_chp.append(dfc.find_value_in_results_df(df, 'co2_chp_construction'))
        list_co2_electricity_substituted.append((dfc.find_value_in_results_df(df, 'co2_electricity_mix')) * -1) # multiply by -1, because they are saved emissions
        list_co2_heat_substituted.append((dfc.find_value_in_results_df(df, 'co2_heat_oil')) * -1)   # multiply by -1, because they are saved emissions

    label_list = ['N\u2082O', 'CH\u2084', 'Transport', 'Electricity', 'Construction AD', 'Construction CHP', 'Electricity substituted', 'Heat substituted']
    data_df = pd.DataFrame({
        'Types': list_types,
        'N\u2082O': list_co2_n2o,
        'CH\u2084': list_co2_ch4,
        'Transport': list_co2_transport,
        'Electricity': list_co2_electricity,
        'Construction AD': list_co2_construction_ad,
        'Construction CHP': list_co2_construction_chp,
        'Electricity substituted': list_co2_electricity_substituted,
        'Heat substituted': list_co2_heat_substituted
    })
    title = 'Global Warming Potential 100'
    y_axis_title = 'GWP 100 [kg CO\u2082 eq.]'

    return dfc.create_bar_chart(label_list, data_df, title, y_axis_title)


def bar_chart_ch4(list_input_df):
    """
    Creates a bar chart of N2O emissions by the different pathways
    :param untreated_df:
    :param ad_df:
    :param steam_df:
    :param biogas_df:
    :param steam_biogas_df:
    :return: bar chart of N2O emissions
    """
    list_types = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

    list_ch4_pre_storage = []
    list_ch4_ad = []
    list_ch4_post_storage = []
    list_ch4_upgrading = []

    for df in list_input_df:
        list_ch4_pre_storage.append(dfc.find_value_in_results_df(df, 'co2_methane_pre_storage'))
        list_ch4_post_storage.append(dfc.find_value_in_results_df(df, 'co2_methane_post_storage'))
        list_ch4_upgrading.append(dfc.find_value_in_results_df(df, 'co2_methane_biogas_upgrading'))
        list_ch4_ad.append(dfc.find_value_in_results_df(df, 'co2_methane_ad'))

    label_list = ['Pre Storage', 'Anaerobic Digestion', 'Post Storage', 'Biogas Upgrading']
    data_df = pd.DataFrame({
        'Types': list_types,
        'Pre Storage': list_ch4_pre_storage,
        'Anaerobic Digestion': list_ch4_ad,
        'Post Storage': list_ch4_post_storage,
        'Biogas Upgrading': list_ch4_upgrading
    })
    title = 'CH\u2084 emissions'
    y_axis_title = 'CH\u2084 emissions [kg CH\u2084]'

    return dfc.create_bar_chart(label_list, data_df, title, y_axis_title)


def bar_chart_ubp(list_input_df):
    """
    Creates a bar chart of UBP by the different pathways
    :param untreated_df:
    :param ad_df:
    :param steam_df:
    :param biogas_df:
    :param steam_biogas_df:
    :return: bar chart of UBP
    """
    list_types = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

    list_co2_emissions = []
    list_nh3_emissions = []
    list_energy = []


    for df in list_input_df:
        list_co2_emissions.append(dfc.find_value_in_results_df(df, 'ubp_co2'))
        list_nh3_emissions.append(dfc.find_value_in_results_df(df, 'ubp_nh3'))
        list_energy.append(dfc.find_value_in_results_df(df, 'ubp_electricity_demand_non_renew'))

    label_list = ['CO\u2082 emissions', 'NH\u2083 emissions', 'Energy']
    data_df = pd.DataFrame({
        'Types': list_types,
        'CO\u2082 emissions': list_co2_emissions,
        'NH\u2083 emissions': list_nh3_emissions,
        'Energy': list_energy,
    })
    title = 'Schweizerische Umweltbelastungspunkte'
    y_axis_title = 'Umweltbelastungspunkte [UBP]'

    return dfc.create_bar_chart(label_list, data_df, title, y_axis_title)


def bar_chart_energy(list_input_df):
    """
    Creates a bar chart of UBP by the different pathways
    :param untreated_df:
    :param ad_df:
    :param steam_df:
    :param biogas_df:
    :param steam_biogas_df:
    :return: bar chart of UBP
    """
    list_types = ['PW 1', 'PW 2', 'PW 3', 'PW 4', 'PW 5']

    list_electricity_generated = []
    list_heat_generated = []


    for df in list_input_df:
        list_electricity_generated.append(dfc.find_value_in_results_df(df, 'electricity_generated_tot'))
        list_heat_generated.append(dfc.find_value_in_results_df(df, 'heat_generated_tot'))


    label_list = ['Electricity', 'Heat']
    data_df = pd.DataFrame({
        'Types': list_types,
        'Electricity': list_electricity_generated,
        'Heat': list_heat_generated,
    })
    title = 'Energy generation'
    y_axis_title = 'Energy [kWh]'

    return dfc.create_bar_chart(label_list, data_df, title, y_axis_title)


def create_bar_chart_list(list_input_df):
    bar_charts_list = [dcc.Graph(id='bar_ch4', figure=bar_chart_ch4(list_input_df)),
                       dcc.Graph(id='bar_co2', figure=bar_chart_co2(list_input_df)),
                       dcc.Graph(id='bar_n2o', figure=bar_chart_n2o(list_input_df)),
                       dcc.Graph(id='bar_nh3', figure=bar_chart_nh3(list_input_df)),
                       dcc.Graph(id='bar_energy', figure=bar_chart_energy(list_input_df)),
                       dcc.Graph(id='bar_ubp', figure=bar_chart_ubp(list_input_df))]

    return bar_charts_list
















