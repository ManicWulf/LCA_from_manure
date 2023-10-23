"""
File for all plots and charts

"""

import pandas as pd

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
                        "Electricity", "Electricity", "Transport"
                        ]
    labels_sources = ["Pre storage", "AD process", "Biogas upgrading", "Post storage", "Field application",
                      "Pre storage", "AD process", "Biogas upgrading", "Post storage", "Field application",
                      "Anaerobic Digestion", "Biogas upgrading",
                      None
                      ]
    """get relevant values from untreated df"""
    # methane sources
    co2_methane_pre_storage = round(dfc.find_value_in_results_df(input_df, "co2_methane_pre_storage"), 2)
    co2_methane_ad = round(dfc.find_value_in_results_df(input_df, "co2_methane_ad"), 2)
    co2_methane_biogas_upgrading = round(dfc.find_value_in_results_df(input_df, "co2_methane_biogas_upgrading"), 2)
    co2_methane_post_storage = round(dfc.find_value_in_results_df(input_df, "co2_methane_post_storage"), 2)
    co2_methane_field = round(dfc.find_value_in_results_df(input_df, "co2_methane_field"), 2)

    #n2o sources
    co2_n2o_pre_storage = round(dfc.find_value_in_results_df(input_df, "co2_n2o_pre_storage"), 2)
    co2_n2o_ad = round(dfc.find_value_in_results_df(input_df, "co2_n2o_ad"), 2)
    co2_n2o_biogas_upgrading = round(dfc.find_value_in_results_df(input_df, "co2_n2o_biogas_upgrading"), 2)
    co2_n2o_post_storage = round(dfc.find_value_in_results_df(input_df, "co2_n2o_post_storage"), 2)
    co2_n2o_field = round(dfc.find_value_in_results_df(input_df, "co2_n2o_field"), 2)

    #electricity demand sources
    co2_electricity_demand_ad = round(dfc.find_value_in_results_df(input_df, "co2_electricity_demand_ad"), 2)
    co2_electricity_demand_biogas_upgrading = round(dfc.find_value_in_results_df(input_df, "co2_electricity_demand_biogas_upgrading"), 2)

    #transport sources
    co2_transport = round(dfc.find_value_in_results_df(input_df, "co2_transport"), 2)

    """create values list"""
    values = [co2_methane_pre_storage, co2_methane_ad, co2_methane_biogas_upgrading, co2_methane_post_storage, co2_methane_field,
              co2_n2o_pre_storage, co2_n2o_ad, co2_n2o_biogas_upgrading, co2_n2o_post_storage, co2_n2o_field,
              co2_electricity_demand_ad, co2_electricity_demand_biogas_upgrading,
              co2_transport]

    return dfc.create_sunburst_chart(labels_emissions, labels_sources, values, title)

















