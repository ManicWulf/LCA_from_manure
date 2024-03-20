"""
Contains all functions to calculate transport emissions


"""
import dash_functions_and_callbacks as dfc


def get_fuel_consumption(env_config):
    return dfc.find_value_env_config("fuel_consumption_tractor", "value", env_config)


def get_co2_diesel(env_config):
    return dfc.find_value_env_config("co2_eq_diesel", "value", env_config)


def fuel_consumption_transport(manure_volume, distance, env_config):
    """
    :param manure_volume: total manure in kg (assuming density of 1)
    :param distance: distance between farm and plant
    :param env_config:
    :return: fuel consumption in l diesel
    """
    """Transport with a 10t lorry, a 1:1 dilution and one trip is the distance * 2 for the return trip"""
    # convert from kg to t
    manure_volume = manure_volume / 1000
    num_trips = (manure_volume / 10) * 4
    fuel_per_trip = distance * get_fuel_consumption(env_config)
    return num_trips * fuel_per_trip


def env_impact_diesel(diesel, env_config):
    return diesel * get_co2_diesel(env_config)




