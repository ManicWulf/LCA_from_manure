"""
file for all environmental impact calculations
"""

import dash_functions_and_callbacks as dfc


""" functions to get values from env config"""


def get_methane_impact(env_config):
    return dfc.find_value_env_config("co2_eq_ch4", "value", env_config)


def get_n2o_impact(env_config):
    return dfc.find_value_env_config("co2_eq_n2o", "value", env_config)


def get_diesel_impact(env_config):
    return dfc.find_value_env_config("co2_eq_diesel", "value", env_config)


def get_ubp_nh3(env_config):
    return dfc.find_value_env_config("ubp_factor_nh3", "value", env_config)


def get_ubp_co2(env_config):
    return dfc.find_value_env_config("ubp_factor_co2", "value", env_config)


def get_ubp_energy_non_renew(env_config):
    return dfc.find_value_env_config("ubp_factor_energy_non_renew", "value", env_config)


def get_ubp_energy_renew(env_config):
    return dfc.find_value_env_config("ubp_factor_energy_renew", "value", env_config)


def get_co2_electricity_mix_che(env_config):
    return dfc.find_value_env_config("co2_eq_el_mix_che", "value", env_config)


def get_co2_heating_oil(env_config):
    return dfc.find_value_env_config("co2_eq_oil_heating", "value", env_config)


def get_co2_heating_gas(env_config):
    return dfc.find_value_env_config("co2_eq_gas_heating", "value", env_config)


def get_co2_construction_ad_plant(env_config):
    return dfc.find_value_env_config("co2_eq_ad_plant_construction", "value", env_config)


def get_co2_construction_chp_generator(env_config):
    return dfc.find_value_env_config("co2_eq_chp_generator_construction", "value", env_config)


def get_daly_el_mix_che(env_config):
    return dfc.find_value_env_config("daly_el_mix_che", "value", env_config)


def get_daly_nh3_che(env_config):
    return dfc.find_value_env_config("nh3_factor_daly_che", "value", env_config)


"""environmental impact functions"""

"""co2 equivalents"""


def co2_methane(env_config, methane_emission):
    """
    :param env_config:
    :param methane_emission: methane emissions in kg methane
    :return: CO2 equivalent for GWP 100 in kg CO2
    """
    return methane_emission * get_methane_impact(env_config)


def co2_n2o(env_config, n2o_emissions):
    """
    :param env_config:
    :param n2o_emissions: emissions in kg N2O
    :return: CO2 equivalent for GWP 100 in kg CO2
    """
    return n2o_emissions * get_n2o_impact(env_config)


"""transportation impact"""


def env_impact_diesel(env_config, diesel):
    """
    :param env_config:
    :param diesel: diesel consumption in l
    :return: env impact in kg CO2 eq.
    """
    return diesel * get_diesel_impact(env_config)


"""heating and electricity impacts"""


def co2_electricity_mix(env_config, electricity):
    """
    :param env_config:
    :param electricity: electricity generated in kWh
    :return: CO2 eq. of the same amount of electricity from the el. Net, in kg CO2
    """
    return electricity * get_co2_electricity_mix_che(env_config)


def co2_heating_oil(env_config, heat):
    """
    :param env_config:
    :param heat: heat generated in kWh
    :return: CO2 eq. of same amount of heat if produced with heating oil, in kg CO2
    """
    return heat * get_co2_heating_oil(env_config)


def daly_electricity_mix(env_config, electricity):
    """
    :param env_config:
    :param electricity: in kwh
    :return: DALYs in days
    """
    daly_factor = get_daly_el_mix_che(env_config)
    return daly_factor * electricity


def daly_nh3_emissions(env_config, nh3):
    """
    :param env_config:
    :param nh3: in kg nh3
    :return: daly in days
    """
    daly_factor = get_daly_nh3_che(env_config)
    return daly_factor * nh3


"""Construction impacts"""

def gwp100_ad_plant_construction(env_config):
    """
    gwp 100 for AD plant construction with a 20 year lifetime
    :param env_config:
    :return:
    """
    lifetime = 20
    return get_co2_construction_ad_plant(env_config) / lifetime


def gwp100_chp_generator_construction(env_config):
    """
    gwp 100 for chp generator construction with a runtime of 80'000 hours
    :param env_config:
    :return:
    """
    lifetime = 10
    return get_co2_construction_chp_generator(env_config) / lifetime


"""Schweizerische Umweltbelastungspunkte (Swiss aggregated impact factor)"""


def ubp_nh3(env_config, nh3_emission):
    return get_ubp_nh3(env_config) * nh3_emission


def ubp_co2_eq(env_config, co2_eq):
    return get_ubp_co2(env_config) * co2_eq


def ubp_energy_non_renew(env_config, energy_non_renew):
    return energy_non_renew * get_ubp_energy_non_renew(env_config)


def ubp_energy_renew(env_config, energy_renew):
    return energy_renew * get_ubp_energy_renew(env_config)



