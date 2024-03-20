"""
Contains all functions in regards to Steam-pretreatment and its effect.
Increase in methane potential, energy costs etc.

"""
import dash_functions_and_callbacks as dfc


def get_methane_increase(env_config):
    return dfc.find_value_env_config("methane_increase_factor", "value", env_config)


def get_heat_demand(env_config):
    return dfc.find_value_env_config("energy_demand_steam", "value", env_config)


def methane_yield_steam(env_config, methane_yield):
    return get_methane_increase(env_config) * methane_yield


def heat_demand_steam(env_config, manure):
    """
    :param env_config:
    :param manure: Solid manure and straw in kg
    :return: heat demand in kWh
    """
    mj_to_kwh = 0.277   # conversion factor for MJ to kWh
    return manure * get_heat_demand(env_config) * mj_to_kwh








