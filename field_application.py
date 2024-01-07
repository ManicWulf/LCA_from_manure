"""
Contains all functions to calculate emissions from field application


"""


import dash_functions_and_callbacks as dfc


"""
get necessary values from env_config file for calculations
"""

def get_acc_nitrogen_factor_manure(env_config):
    "accessible nitrogen in manure"
    return dfc.find_value_env_config("factor_n_acc_manure", "value", env_config)


def get_acc_nitrogen_factor_digestate(env_config):
    "accessible nitrogen in digestate"
    return dfc.find_value_env_config("factor_n_acc_digestate", "value", env_config)


def get_nh3_field_emissions_factor(env_config):
    return dfc.find_value_env_config("factor_nh3_emission_field", "value", env_config)


def get_n2o_field_emissions_factor(env_config):
    return dfc.find_value_env_config("factor_n2o_emission_field", "value", env_config)


def get_field_application_factor(env_config, application_method):
    """get the field application factor, for reducing emissions based on application method"""
    if 'shoe' in application_method:
        return dfc.find_value_env_config("factor_field_application_trailing_shoe", "value", env_config)
    elif 'hose' in application_method:
        return dfc.find_value_env_config("factor_field_application_trailing_hose", "value", env_config)
    else:
        return dfc.find_value_env_config("factor_field_application_splash_plate", "value", env_config)


"""
functions to calculate field emissions
"""


def nh3_emissions_manure(env_config, n_tot, application_method):
    n_acc = n_tot * get_acc_nitrogen_factor_manure(env_config)
    nh3_emissions = n_acc * get_nh3_field_emissions_factor(env_config) * get_field_application_factor(env_config, application_method)
    return n_acc, nh3_emissions


def nh3_emissions_digestate(env_config, n_tot, application_method):
    n_acc = n_tot * get_acc_nitrogen_factor_digestate(env_config)
    nh3_emissions = n_acc * get_nh3_field_emissions_factor(env_config) * get_field_application_factor(env_config, application_method)
    return n_acc, nh3_emissions


def n2o_emissions(env_config, n_acc, nh3_emissions, application_method):
    """
    n2o emissions calculation does not differ between digestate and manure, since only the n_acc is affected by that
    """
    n_new = n_acc - nh3_emissions   #adjust n_acc by already emitted NH3 emissions
    """
    Need to check if the field application factor is applied here as well
    But it would make sense to do so, otherwise this field application method would lead to increased N2O emissions!!
    """
    n2o_emission = n_new * get_n2o_field_emissions_factor(env_config) * get_field_application_factor(env_config, application_method)
    return n2o_emission


def n_emissions_manure(env_config, n_tot, application_method):
    n_acc, nh3_emissions = nh3_emissions_manure(env_config, n_tot, application_method)
    n2o_emission = n2o_emissions(env_config, n_acc, nh3_emissions, application_method)
    return nh3_emissions, n2o_emission


def n_emissions_digestate(env_config, n_tot, application_method):
    n_acc, nh3_emissions = nh3_emissions_digestate(env_config, n_tot, application_method)
    n2o_emission = n2o_emissions(env_config, n_acc, nh3_emissions, application_method)
    return nh3_emissions, n2o_emission












