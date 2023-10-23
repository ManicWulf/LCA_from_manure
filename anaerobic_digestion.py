"""
Contains all functions regarding the AD process.
Emissions, biogas production etc.

"""

import dash_functions_and_callbacks as dfc

"""
functions to get all necessary values from env config
"""


def get_thermal_value_methane(env_config):
    return dfc.find_value_env_config("tv_methane", "value", env_config)


def get_seconds_per_year(env_config):
    return dfc.find_value_env_config("seconds_per_year", "value", env_config)


def get_biogas_composition(env_config):
    ch4_content = dfc.find_value_env_config("methane_biogas", "value", env_config)
    co2_content = dfc.find_value_env_config("co2_biogas", "value", env_config)
    return ch4_content, co2_content


def get_methane_loss(env_config):
    return dfc.find_value_env_config("ch4_loss_ad", "value", env_config)


def get_heat_demand(env_config):
    return dfc.find_value_env_config("heat_demand_ad", "value", env_config)


def get_electricity_demand(env_config):
    return dfc.find_value_env_config("electricity_demand_ad", "value", env_config)


def get_plant_efficiency(env_config, size):
    """
    :param env_config: environmental config file
    :param size: size of the plant, 1 for small, 2 for medium, 3 for large
    :return: methane yield efficiency of a plant of corresponding size
    """
    if size == 1:       #small plant size
        return dfc.find_value_env_config("methane_yield_efficiency_small", "value", env_config)
    elif size == 2:      #medium plant size
        return dfc.find_value_env_config("methane_yield_efficiency_medium", "value", env_config)
    elif size == 3:     #large plant size
        return dfc.find_value_env_config("methane_yield_efficiency_large", "value", env_config)


"""
functions to calculate the AD process
"""



def plant_size(env_config, bmp):
    """
    :param env_config: environmental config file
    :param bmp: theoretical bio methane potention
    :return: plant size based on the power potential of the plant, under the assumption the plant runs with the
    lowest efficiency of a small plant. This is done, so that plant size is rounded down rather than up
    """
    efficiency = get_plant_efficiency(env_config, 1)
    power = bmp * efficiency * get_thermal_value_methane(env_config) / get_seconds_per_year(env_config)
    if power <= 30:
        """plant size is small"""
        return 1
    elif (power > 30) & (power <= 150):
        """plant size is medium"""
        return 2
    else:
        """plant size is large"""
        return 3


def calc_methane_yield(env_config, bmp):
    """
    :param env_config:
    :param bmp: theoretical biomethane potential
    :return: methane yield of the plant based on it's size
    """
    return bmp * get_plant_efficiency(env_config, plant_size(env_config, bmp))


def biogas_composition(env_config, methane_yield):
    """calculate the composition of the biogas, namely CO2 content and total biogas based on methane yield"""
    ch4_factor, co2_factor = get_biogas_composition(env_config)
    biogas_tot = methane_yield / ch4_factor
    co2_content = biogas_tot * co2_factor
    return biogas_tot, co2_content


def calc_methane_loss(env_config, methane_yield):
    return get_methane_loss(env_config) * methane_yield


def calc_heat_demand_ad(env_config, biogas):
    """
    :param env_config: env_config file
    :param biogas: total biogas produced in m3
    :return: heat demand in kWh
    """
    return biogas * get_heat_demand(env_config) / 3600


def calc_electricity_demand_ad(env_config, biogas):
    """
    :param env_config:
    :param biogas: total biogas produced in m3
    :return: electricity demand in kWh
    """
    return biogas * get_electricity_demand(env_config)


