"""
calculations regarding the co-heat-and-power generator (CHP)
"""

import dash_functions_and_callbacks as dfc


def get_chp_efficiency_heat(env_config):
    return dfc.find_value_env_config("chp_heat_efficiency", "value", env_config)


def get_chp_efficiency_electricity(env_config):
    return dfc.find_value_env_config("chp_electric_efficiency", "value", env_config)


def get_thermal_value_methane(env_config):
    return dfc.find_value_env_config("tv_methane", "value", env_config)


def get_seconds_per_year(env_config):
    return dfc.find_value_env_config("seconds_per_year", "value", env_config)


def get_hours_per_year(env_config):
    return dfc.find_value_env_config("hours_per_year", "value", env_config)


def get_operating_time_chp(env_config):
    return dfc.find_value_env_config("operating_time_chp", "value", env_config)


def get_efficiency_sofc(env_config):
    return dfc.find_value_env_config("efficiency_el_sofc", "value", env_config)


def energy_produced(eff_methane, env_config):       #in kWh
    tvm = get_thermal_value_methane(env_config)
    s_in_y = get_seconds_per_year(env_config)
    h_in_y = get_hours_per_year(env_config)
    operating_time = get_operating_time_chp(env_config)
    factor_operating_time = h_in_y / (s_in_y * operating_time)
    heat = eff_methane * tvm * get_chp_efficiency_heat(env_config) * factor_operating_time * 1000
    electricity = eff_methane * tvm * get_chp_efficiency_electricity(env_config) * factor_operating_time * 1000
    return heat, electricity


def electricity_generated_sofc(env_config, biomethane):
    tvm = get_thermal_value_methane(env_config)
    s_in_y = get_seconds_per_year(env_config)
    h_in_y = get_hours_per_year(env_config)
    operating_time = get_operating_time_chp(env_config)
    factor_operating_time = h_in_y / (s_in_y * operating_time)
    return biomethane * tvm * factor_operating_time * get_efficiency_sofc(env_config) * 1000

