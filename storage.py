"""
Contains all functions to calculate Pre- and Post-Storage emissions


"""

import dash_functions_and_callbacks as dfc

"""
recursive function to calculate storage emissions based on time and a daily emissions factor, in 1 day increments
"""
def storage_emissions(time, x_tot, emission_factor):
    """
    :param time: storage time in days
    :param x_tot: total nutrient value before storage in kg elemental (N, C)
    :param emission_factor: nutrient specific daily emission value
    :return: recursive function, returns emissions in kg elemental (N, C)
    """
    x_tot_new = x_tot * ((1 - emission_factor)**time)
    emission = x_tot - x_tot_new
    """emission = x_tot * emission_factor
    x_tot_new = x_tot - emission
    if time > 0 and x_tot_new > 0:
        return emission + storage_emissions(time - 1, x_tot_new, emission_factor)"""
    return emission


"""
get emissions factor for specific storage emissions: 
emission_factor_nh3_storage_digestate_daily
emission_factor_n2o_storage_digestate_daily
emission_factor_ch4_storage_digestate_daily
emission_factor_nh3_storage_manure_daily
emission_factor_n2o_storage_manure_daily
emission_factor_ch4_storage_manure_daily

"""


def get_emissions_factor_digestate_nh3(env_config):
    return dfc.find_value_env_config("emission_factor_nh3_storage_digestate_daily", "value", env_config)


def get_emissions_factor_manure_nh3(env_config):
    return dfc.find_value_env_config("emission_factor_nh3_storage_manure_daily", "value", env_config)


def get_emissions_factor_digestate_n2o(env_config):
    return dfc.find_value_env_config("emission_factor_n2o_storage_digestate_daily", "value", env_config)


def get_emissions_factor_manure_n2o(env_config):
    return dfc.find_value_env_config("emission_factor_n2o_storage_manure_daily", "value", env_config)


def get_emissions_factor_digestate_ch4(env_config):
    return dfc.find_value_env_config("emission_factor_ch4_storage_digestate_daily", "value", env_config)


def get_emissions_factor_manure_ch4(env_config):
    return dfc.find_value_env_config("emission_factor_ch4_storage_manure_daily", "value", env_config)


"""
calculate storage emissions
"""


def emissions_manure_ch4(time, x_tot, env_config):
    emissions_factor = get_emissions_factor_manure_ch4(env_config)
    return storage_emissions(time, x_tot, emissions_factor)


def emissions_digestate_ch4(time, x_tot, env_config):
    emissions_factor = get_emissions_factor_digestate_ch4(env_config)
    return storage_emissions(time, x_tot, emissions_factor)


def emissions_manure_n2o(time, x_tot, env_config):
    emissions_factor = get_emissions_factor_manure_n2o(env_config)
    return storage_emissions(time, x_tot, emissions_factor)


def emissions_digestate_n2o(time, x_tot, env_config):
    emissions_factor = get_emissions_factor_digestate_n2o(env_config)
    return storage_emissions(time, x_tot, emissions_factor)


def emissions_manure_nh3(time, x_tot, env_config):
    emissions_factor = get_emissions_factor_manure_nh3(env_config)
    return storage_emissions(time, x_tot, emissions_factor)


def emissions_digestate_nh3(time, x_tot, env_config):
    emissions_factor = get_emissions_factor_digestate_nh3(env_config)
    return storage_emissions(time, x_tot, emissions_factor)

