"""
Contains all functions to calculate the individual farms values, such as:
total manure (seperated in solid, liquid and straw), total methane potential, total Nitrogen


"""
import logging

import anaerobic_digestion as ad
import dash_functions_and_callbacks as dfc
import storage
import transport
import field_application as field
import co_heat_power_generator as chp
import biogas_upgrading as bg
import steam
import environmental_impacts as envi


# calculates the collection rate of manure based on time spent outside of stable
def manure_collection_rate(hours, days):
    rate = ((24 - hours) / 24) * ((365 - days) / 365)
    return rate


"""
calculate total carbon based on theoretical methane potential. 
Since we only want this to calculate emissions, we only need carbon from theoretic methane potential.
Other carbon would not be emitted as methane anyways.

"""


def total_carbon(methane_pot, env_config):
    """
    :param methane_pot: methane potential in m3 methane
    :param env_config:
    :return: total carbon from that methane potential in kg C
    """
    densitity_methane = dfc.find_value_env_config("density_ch4", "value", env_config)
    molar_ratio_c_ch4 = dfc.find_value_env_config("molar_ratio_c_to_ch4", "value", env_config)
    methane_mass = densitity_methane * methane_pot
    mass_c = methane_mass * molar_ratio_c_ch4
    return mass_c


def carbon_to_methane_volume(carbon, env_config):
    """
    :param carbon: methane in kg C that needs to be converted to m3
    :param env_config:
    :return: methane in m3
    """
    densitity_methane = dfc.find_value_env_config("density_ch4", "value", env_config)
    molar_ratio_c_ch4 = dfc.find_value_env_config("molar_ratio_c_to_ch4", "value", env_config)
    methane_mass = carbon / molar_ratio_c_ch4
    methane_volume = methane_mass / densitity_methane
    return methane_volume


def methane_mass_to_volume(env_config, methane_mass):
    """
    :param env_config:
    :param methane_mass:
    :return:
    """
    densitity_methane = dfc.find_value_env_config("density_ch4", "value", env_config)
    return methane_mass / densitity_methane


def methane_volume_to_mass(env_config, methane_volume):
    """
    :param env_config:
    :param methane_volume:
    :return:
    """
    densitity_methane = dfc.find_value_env_config("density_ch4", "value", env_config)
    return methane_volume * densitity_methane


def methane_to_carbon(env_config, methane):
    """
    :param env_config:
    :param methane: methane in kg CH4
    :return: methane in kg C
    """
    molar_ratio_c_ch4 = dfc.find_value_env_config("molar_ratio_c_to_ch4", "value", env_config)
    return methane * molar_ratio_c_ch4


def carbon_to_methane(env_config, carbon):
    """
    :param env_config:
    :param carbon: methane in kg C
    :return: methane in kg CH4
    """
    molar_ratio_c_ch4 = dfc.find_value_env_config("molar_ratio_c_to_ch4", "value", env_config)
    return carbon / molar_ratio_c_ch4


def n2o_to_n(env_config, n2o):
    """
    :param env_config:
    :param n2o: n2o in kg N2O
    :return: n2o in kg N
    """
    molar_ratio_n_n2o = dfc.find_value_env_config("molar_ratio_n_to_n2o", "value", env_config)
    return n2o * molar_ratio_n_n2o


def n_to_n2o(env_config, n):
    """
    :param env_config:
    :param n: N2O in kg N
    :return: N2O in kg N2O
    """
    molar_ratio_n_n2o = dfc.find_value_env_config("molar_ratio_n_to_n2o", "value", env_config)
    return n / molar_ratio_n_n2o


def nh3_to_n(env_config, nh3):
    """
    :param env_config:
    :param nh3: NH3 in kg NH3
    :return: NH3 in kg N
    """
    molar_ratio_n_nh3 = dfc.find_value_env_config("molar_ratio_n_to_nh3", "value", env_config)
    return nh3 * molar_ratio_n_nh3


def n_to_nh3(env_config, n):
    """
    :param env_config:
    :param n: NH3 in kg N
    :return: NH3 in kg NH3
    """
    molar_ratio_n_nh3 = dfc.find_value_env_config("molar_ratio_n_to_nh3", "value", env_config)
    return n / molar_ratio_n_nh3


"""
takes farm_data, and animal_config and env config Dataframes
calculates manure, methane potential, pre-storage emissions and transportation costs
Caluclations for 1 farm file!
"""

# list of all results from single farm calculations
results_single_farm = ["manure_solid", "manure_liquid", "manure_straw", "methane_solid", "methane_liquid",
                       "methane_straw", "methane_tot", "ch4_emissions", "nh3_emissions", "n2o_emissions", "n_tot",
                       "n_tot_pre_storage",
                       "transport_emissions", "sum_post_storage_time", "c_tot", "c_tot_pre_storage", "c_tot_solid",
                       "c_tot_liquid", "c_tot_straw"]


def get_animal_data(animal, farm_data):
    """
    Get the animal data and store it in a dictionary
    :param animal: type of animal
    :param farm_data: the input data from the user
    :return: Return a dictionary with relevant animal data
    """
    animal_data_dict = {'days_outside': dfc.find_value_farm(animal, "days-outside", farm_data),
                        'hours_outside': dfc.find_value_farm(animal, "hours-outside", farm_data),
                        'manure_type': dfc.find_value_farm(animal, "manure-type", farm_data),
                        'num_animal': dfc.find_value_farm(animal, "num-animals", farm_data)}

    # manure collection factor based on time spent outside of stable
    animal_data_dict['manure_collection'] = manure_collection_rate(animal_data_dict['hours_outside'],
                                                                   animal_data_dict['days_outside'])
    return animal_data_dict


def get_methane_potentials(animal, animal_config):
    """
    Retrieves the methane potentials for liquid, solid, and straw manure types for a given animal.

    :param animal: The type of animal to retrieve data for.
    :param animal_config: The configuration containing animal data.
    :return: A dictionary containing methane potentials for liquid, solid, and straw.
    """
    methane_pot_dict = {
        'methane_pot_liquid': dfc.find_value_animal_config_1_variable(animal, "methane_potential_liquid",
                                                                      animal_config),
        'methane_pot_solid': dfc.find_value_animal_config_1_variable(animal, "methane_potential_solid", animal_config),
        'methane_pot_straw': dfc.find_value_animal_config_1_variable(animal, "methane_potential_straw", animal_config)}
    return methane_pot_dict


def get_dw_odw(animal, animal_config):
    """
    Retrieves the dry weight, organic dry weight and VS of manure and straw for given animal type.

    :param animal: The type of animal to retrieve data for.
    :param animal_config: The configuration containing animal data.
    :return: A dictionary containing dw, odw and vs of manure and straw.
    """
    # get dry weight and organic dry weight for manure, and vs for straw
    dw_odw_dict = {'dw_liquid': dfc.find_value_animal_config_1_variable(animal, "dry_weight_liquid", animal_config),
                   'odw_liquid': dfc.find_value_animal_config_1_variable(animal, "organic_dry_weight_liquid",
                                                                         animal_config),
                   'dw_solid': dfc.find_value_animal_config_1_variable(animal, "dry_weight_solid", animal_config),
                   'odw_solid': dfc.find_value_animal_config_1_variable(animal, "dry_weight_solid", animal_config),
                   'vs_straw': dfc.find_value_animal_config_1_variable(animal, "vs_content_straw", animal_config)}

    return dw_odw_dict


def calc_farm_methane_pot(manure_methane_dict, dw_odw_dict, methane_pot_dict):
    # calculate methane potential of the farm
    manure_methane_dict['methane_solid'] = manure_methane_dict['manure_solid'] * dw_odw_dict['dw_solid'] * dw_odw_dict[
        'odw_solid'] * \
                                           methane_pot_dict['methane_pot_solid']
    manure_methane_dict['methane_liquid'] = manure_methane_dict['manure_liquid'] * dw_odw_dict['dw_liquid'] * \
                                            dw_odw_dict['odw_liquid'] * \
                                            methane_pot_dict['methane_pot_liquid']
    manure_methane_dict['methane_straw'] = manure_methane_dict['manure_straw'] * dw_odw_dict['vs_straw'] * \
                                           methane_pot_dict[
                                               'methane_pot_straw']
    return manure_methane_dict


def calc_manure_and_methane_cattle(animal, animal_data_dict, animal_config, methane_pot_dict, dw_odw_dict):
    # get manure total
    manure_methane_dict = {}
    manure_pot_liquid = dfc.find_value_animal_config(animal, animal_data_dict['manure_type'], "manure_liquid",
                                                     animal_config)
    manure_pot_solid = dfc.find_value_animal_config(animal, animal_data_dict['manure_type'], "manure_solid",
                                                    animal_config)
    manure_pot_straw = dfc.find_value_animal_config(animal, animal_data_dict['manure_type'], "manure_straw",
                                                    animal_config)
    manure_methane_dict['manure_solid'] = manure_pot_solid * animal_data_dict['num_animal'] * animal_data_dict[
        'manure_collection']
    manure_methane_dict['manure_liquid'] = manure_pot_liquid * animal_data_dict['num_animal'] * animal_data_dict[
        'manure_collection']
    manure_methane_dict['manure_straw'] = manure_pot_straw * animal_data_dict['num_animal'] * animal_data_dict[
        'manure_collection']

    # calculate methane potential of the farm
    manure_methane_dict = calc_farm_methane_pot(manure_methane_dict, dw_odw_dict, methane_pot_dict)

    return manure_methane_dict


def calc_manure_and_methane_pig_poultry(animal, animal_data_dict, animal_config, methane_pot_dict, dw_odw_dict):
    manure_methane_dict = {}
    # get manure total
    manure_pot_liquid = dfc.find_value_animal_config_1_variable(animal, "manure_liquid", animal_config)
    manure_pot_solid = dfc.find_value_animal_config_1_variable(animal, "manure_solid", animal_config)
    manure_pot_straw = dfc.find_value_animal_config_1_variable(animal, "manure_straw", animal_config)
    manure_methane_dict['manure_solid'] = manure_pot_solid * animal_data_dict['num_animal'] * animal_data_dict[
        'manure_collection']
    manure_methane_dict['manure_liquid'] = manure_pot_liquid * animal_data_dict['num_animal'] * animal_data_dict[
        'manure_collection']
    manure_methane_dict['manure_straw'] = manure_pot_straw * animal_data_dict['num_animal'] * animal_data_dict[
        'manure_collection']

    # calculate methane potential of the farm
    manure_methane_dict = calc_farm_methane_pot(manure_methane_dict, dw_odw_dict, methane_pot_dict)
    return manure_methane_dict


def add_dict_to_results(results_df, new_dict):
    for key, value in new_dict.items():
        dfc.add_value_to_results_df(results_df, key, "value", value)


def store_dict_in_results(results_df, new_dict):
    for key, value in new_dict.items():
        dfc.store_value_in_results_df(results_df, key, "value", value)


def get_additional_data(farm_data):
    """
    Gets additional data from farm data
    :param farm_data:
    :return:
    """
    additional_data_dict = {'pre_storage': dfc.find_value_farm("pre_storage", "additional-data", farm_data),
                            'post_storage': dfc.find_value_farm("post_storage", "additional-data", farm_data),
                            'distance': dfc.find_value_farm("distance", "additional-data", farm_data)}
    return additional_data_dict


def calc_tot_methane(results_df):
    """get methane values from df and divide by 1000 to convert from NL to m3"""
    methane_dict = {'methane_solid': dfc.find_value_in_results_df(results_df, "methane_solid") / 1000,
                    'methane_liquid': dfc.find_value_in_results_df(results_df, "methane_liquid") / 1000,
                    'methane_straw': dfc.find_value_in_results_df(results_df, "methane_straw") / 1000}

    methane_dict['methane_tot'] = methane_dict['methane_solid'] + methane_dict['methane_liquid'] + methane_dict[
        'methane_straw']
    return methane_dict


def methane_pre_storage_emissions(carbon_emissions_dict, methane_dict, env_config, additional_data_dict):
    # methane pre-storage emissions by source (liquid, solid, straw)
    carbon_emissions_dict['c_tot_solid'] = total_carbon(methane_dict['methane_solid'], env_config)
    carbon_emissions_dict['c_tot_liquid'] = total_carbon(methane_dict['methane_liquid'], env_config)
    carbon_emissions_dict['c_tot_straw'] = total_carbon(methane_dict['methane_straw'], env_config)
    carbon_emissions_dict['methane_emissions_solid'] = storage.emissions_manure_ch4(additional_data_dict['pre_storage'],
                                                                                carbon_emissions_dict['c_tot_solid'],
                                                                                env_config)
    carbon_emissions_dict['methane_emissions_liquid'] = storage.emissions_manure_ch4(additional_data_dict['pre_storage'],
                                                                                 carbon_emissions_dict['c_tot_liquid'],
                                                                                 env_config)
    carbon_emissions_dict['methane_emissions_straw'] = storage.emissions_manure_ch4(additional_data_dict['pre_storage'],
                                                                                carbon_emissions_dict['c_tot_straw'],
                                                                                env_config)

    return carbon_emissions_dict


def convert_ch4_emissions_to_volume(carbon_emissions_dict, env_config):
    """convert the emissions back to volume in m3"""
    list_names = ['methane_emissions_solid', 'methane_emissions_liquid', 'methane_emissions_straw']
    for name in list_names:
        new_key = name + '_volume'
        carbon_emissions_dict[new_key] = carbon_to_methane_volume(carbon_emissions_dict[name], env_config)

    return carbon_emissions_dict


def calc_pre_storage_emissions_carbon(methane_dict, env_config, additional_data_dict):
    # pre storage emissions ch4 in kg C
    carbon_emissions_dict = {'c_tot': total_carbon(methane_dict['methane_tot'], env_config)}
    carbon_emissions_dict['methane_emissions_pre_storage'] = storage.emissions_manure_ch4(
        additional_data_dict['pre_storage'], carbon_emissions_dict['c_tot'], env_config)
    carbon_emissions_dict['c_tot_pre_storage'] = carbon_emissions_dict['c_tot'] - carbon_emissions_dict[
        'methane_emissions_pre_storage']

    # after subtracting the CH4-C emissions from c_tot, turn them into kg CH4
    carbon_emissions_dict['methane_emissions_pre_storage'] = carbon_to_methane(env_config, carbon_emissions_dict[
        'methane_emissions_pre_storage'])

    # methane pre-storage emissions by source (liquid, solid, straw)
    carbon_emissions_dict = methane_pre_storage_emissions(carbon_emissions_dict, methane_dict, env_config,
                                                          additional_data_dict)

    """convert the emissions back to volume in m3"""
    carbon_emissions_dict = convert_ch4_emissions_to_volume(carbon_emissions_dict, env_config)

    """calculate methane_pot for total, solid, liquid and straw separately after pre-storage in m3"""
    carbon_emissions_dict['methane_solid_pre_storage'] = methane_dict['methane_solid'] - carbon_emissions_dict[
        'methane_emissions_solid_volume']
    carbon_emissions_dict['methane_straw_pre_storage'] = methane_dict['methane_straw'] - carbon_emissions_dict[
        'methane_emissions_straw_volume']
    carbon_emissions_dict['methane_liquid_pre_storage'] = methane_dict['methane_liquid'] - carbon_emissions_dict[
        'methane_emissions_liquid_volume']
    carbon_emissions_dict['methane_tot_pre_storage'] = carbon_emissions_dict['methane_solid_pre_storage'] + \
                                                       carbon_emissions_dict['methane_liquid_pre_storage'] + \
                                                       carbon_emissions_dict['methane_straw_pre_storage']

    return carbon_emissions_dict


def calc_pre_storage_emissions_nitrogen(nitrogen_emissions_dict, env_config, additional_data_dict):
    """
    pre-storage emissions n2o and nh3
    nh3 first, then subtract nh3 emissions from n_tot for n2o emissions
    emissions in kg N
    """

    nitrogen_emissions_dict['nh3_emissions_pre_storage'] = storage.emissions_manure_nh3(
        additional_data_dict['pre_storage'], nitrogen_emissions_dict['n_tot'], env_config)

    n_tot_new = nitrogen_emissions_dict['n_tot'] - nitrogen_emissions_dict['nh3_emissions_pre_storage']
    nitrogen_emissions_dict['n2o_emissions_pre_storage'] = storage.emissions_manure_n2o(
        additional_data_dict['pre_storage'], n_tot_new, env_config)

    nitrogen_emissions_dict['n_tot_pre_storage'] = n_tot_new - nitrogen_emissions_dict['n2o_emissions_pre_storage']

    # after adjusting n_tot_pre_storage, turn nh3 and n2o emissions into kg NH3 and kg N2O
    nitrogen_emissions_dict['nh3_emissions_pre_storage'] = n_to_nh3(env_config, nitrogen_emissions_dict[
        'nh3_emissions_pre_storage'])
    nitrogen_emissions_dict['n2o_emissions_pre_storage'] = n_to_n2o(env_config, nitrogen_emissions_dict[
        'n2o_emissions_pre_storage'])

    return nitrogen_emissions_dict


def add_total_emissions_pre_storage(results_df, carbon_emissions_dict, nitrogen_emissions_dict):
    dfc.add_value_to_results_df(results_df, "methane_emissions", "value",
                                carbon_emissions_dict['methane_emissions_pre_storage'])
    dfc.add_value_to_results_df(results_df, "nh3_emissions", "value",
                                nitrogen_emissions_dict['nh3_emissions_pre_storage'])
    dfc.add_value_to_results_df(results_df, "n2o_emissions", "value",
                                nitrogen_emissions_dict['n2o_emissions_pre_storage'])


def get_initial_data_field_application(input_df, ad):
    if ad:
        c_tot = dfc.find_value_in_results_df(input_df, "c_tot_ad")
    else:
        c_tot = dfc.find_value_in_results_df(input_df, "c_tot")
    n_tot = dfc.find_value_in_results_df(input_df, "n_tot")
    post_storage_time = dfc.find_value_in_results_df(input_df, "sum_post_storage_time")

    return c_tot, n_tot, post_storage_time


def calc_post_storage_emissions(post_storage_time, c_tot, n_tot, env_config, ad):
    emissions_dict = {}
    if ad:  # check if ad is True, then calculate for digestate
        emissions_dict['methane_emissions_storage'] = storage.emissions_digestate_ch4(post_storage_time, c_tot, env_config)
        emissions_dict['nh3_emissions_storage'] = storage.emissions_digestate_nh3(post_storage_time, n_tot, env_config)

    else:  # if ad is False, calculate for manure
        emissions_dict['methane_emissions_storage'] = storage.emissions_manure_ch4(post_storage_time, c_tot, env_config)
        emissions_dict['nh3_emissions_storage'] = storage.emissions_manure_nh3(post_storage_time, n_tot, env_config)

    # calculate new N after nh3 emissions
    n_tot_new = n_tot - emissions_dict['nh3_emissions_storage']
    # use new value of N to calculate N2O emissions
    if ad:
        emissions_dict['n2o_emissions_storage'] = storage.emissions_digestate_n2o(post_storage_time, n_tot_new,
                                                                                  env_config)

    else:
        emissions_dict['n2o_emissions_storage'] = storage.emissions_manure_n2o(post_storage_time, n_tot_new, env_config)

    return emissions_dict, n_tot_new


def store_post_storage_and_field_emissions(input_df, emissions_dict):
    for key, value in emissions_dict.items():
        suffix = key.rpartition('_')[2]
        name = key.rpartition('_')[0]
        if suffix == 'storage':  # if it's storage emissions, store it as post_storage
            dfc.store_value_in_results_df(input_df, f"{name}_post_storage", "value", value)

        else:  # if it's field emissions, store it as such
            dfc.store_value_in_results_df(input_df, key, "value", value)


def add_post_storage_and_field_emissions(input_df, emissions_dict):
    """add all emissions to total emissions"""
    for key, value in emissions_dict.items():
        new_key = key.rpartition('_')[0]
        dfc.add_value_to_results_df(input_df, new_key, "value", value)


def convert_elemental_to_molecule_weight(input_dict, env_config):
    """before storing, convert all emissions from kg C/N to kg CH4/NH3/N2O"""
    for key, value in input_dict.items():
        if 'nh3' in key:
            input_dict[key] = n_to_nh3(env_config, input_dict[key])

        elif 'n2o' in key:
            input_dict[key] = n_to_n2o(env_config, input_dict[key])

        elif "ch4" in key or 'methane' in key:
            input_dict[key] = carbon_to_methane(env_config, input_dict[key])

    return input_dict


def get_initial_data(input_df, list_data):
    initial_data_dict = {}
    for name in list_data:
        initial_data_dict[name] = dfc.find_value_in_results_df(input_df, name)

    return initial_data_dict


def get_initial_data_env_impact(input_df):
    """get necessary data points from input df"""
    list_names = ["methane_emissions_pre_storage", "methane_emissions_post_storage", "methane_emissions_ad",
                  "methane_emissions_biogas_upgrading",
                  "methane_emissions", "n2o_emissions_pre_storage", "n2o_emissions_post_storage", "n2o_emissions_field",
                  "n2o_emissions", 'nh3_emissions', "electricity_demand_ad", "electricity_demand_biogas_upgrading",
                  "electricity_demand_tot", "electricity_generated_tot", "heat_generated_tot", 'co2_transport']
    initial_data_dict = get_initial_data(input_df, list_names)

    return initial_data_dict


def calc_gwp_impact(initial_data_dict, env_config):
    env_impact_dict = {}
    for key, value in initial_data_dict.items():
        if 'ch4' in key or 'methane' in key:
            # remove "_emissions" from the key
            new_key = key.replace('_emissions', '')
            if new_key == "methane":
                new_key = "methane_tot"
            if 'ch4' in new_key:
                new_key = new_key.replace('ch4', 'methane')

            env_impact_dict[f'co2_{new_key}'] = envi.co2_methane(env_config, value)

        elif 'n2o' in key:
            # remove "_emissions" from the key
            new_key = key.replace('_emissions', '')
            if new_key == "n2o":
                new_key = "n2o_tot"
            env_impact_dict[f'co2_{new_key}'] = envi.co2_n2o(env_config, value)

        elif 'electricity_demand' in key:
            env_impact_dict[f'co2_{key}'] = envi.co2_electricity_mix(env_config, value)

        elif 'electricity_generated_tot' in key:
            env_impact_dict['co2_electricity_mix'] = envi.co2_electricity_mix(env_config, value)

        elif "heat_generated_tot" in key:
            env_impact_dict['co2_heat_oil'] = envi.co2_heating_oil(env_config, value)

    return env_impact_dict


def steam_initial_data(input_df):
    list_data = ["manure_solid", "manure_straw", "methane_solid_pre_storage", "methane_straw_pre_storage",
                 "methane_liquid_pre_storage"]
    steam_initial_data_dict = get_initial_data(input_df, list_data)

    return steam_initial_data_dict


def calc_single_farm(results_df, farm_data, animal_config, env_config):
    nitrogen_emissions_dict = {'n_tot': 0}

    manure_tot = 0

    # get a list of all animal types used from the config file (Attention: Right now it still uses animal_type_ger until Data_input.py is updated!!
    animal_types = dfc.get_animal_types(animal_config)
    logging.debug(f'list of animal types: {animal_types}')
    for animal in animal_types:

        # get values from the farm file
        """create a log for debugging purposes"""
        logging.debug(f"Farm Data: {farm_data.head()}")

        animal_data_dict = get_animal_data(animal, farm_data)

        logging.debug(f"Animal_data_dict for {animal}: {animal_data_dict}")

        # get values from animal_config
        # check if there are any animals in the first place
        if animal_data_dict['num_animal'] > 0:

            # get methane potential

            methane_pot_dict = get_methane_potentials(animal, animal_config)

            # get dry weight and organic dry weight for manure, and vs for straw
            dw_odw_dict = get_dw_odw(animal, animal_config)

            logging.debug(f'Methane_pot_dict for {animal}: {methane_pot_dict}')
            logging.debug(f'dw_odw_dict for {animal}: {dw_odw_dict}')

            # get Nitrogen
            n_tot_per_animal = dfc.find_value_animal_config_1_variable(animal, "nitrogen_content", animal_config)
            nitrogen_emissions_dict['n_tot'] += n_tot_per_animal * animal_data_dict['num_animal'] * animal_data_dict[
                'manure_collection']

            # check if manure type exists, if it doesn't it's probably a Pig or Poultry animal with a wrong input.
            if dfc.find_value_animal_config(animal, animal_data_dict['manure_type'], "animal_type", animal_config):

                # get manure total
                manure_methane_dict = calc_manure_and_methane_cattle(animal, animal_data_dict, animal_config,
                                                                     methane_pot_dict, dw_odw_dict)

                manure_tot += (manure_methane_dict['manure_solid'] + manure_methane_dict['manure_liquid'] +
                               manure_methane_dict['manure_straw'])

                logging.debug(f'manure_methane_dict for {animal}: {manure_methane_dict}')

                # add up methane potential and total manure
                add_dict_to_results(results_df, manure_methane_dict)

            # if the manure type doesn't exist for that animal, assume it's pig or poultry, which only have 1 manure type each and calculate with that
            else:
                # get manure total
                manure_methane_dict = calc_manure_and_methane_pig_poultry(animal, animal_data_dict, animal_config,
                                                                          methane_pot_dict, dw_odw_dict)

                manure_tot += (manure_methane_dict['manure_solid'] + manure_methane_dict['manure_liquid'] +
                               manure_methane_dict['manure_straw'])

                logging.debug(f'manure_methane_dict for {animal}: {manure_methane_dict}')

                # add up methane potential and total manure
                add_dict_to_results(results_df, manure_methane_dict)

        # if the number of animals is 0, continue in the for loop to the next animal
        else:
            continue

    # get additional data (storage durations and transport distance)
    additional_data_dict = get_additional_data(farm_data)

    """get methane values from df and divide by 1000 to convert from NL to m3"""
    methane_dict = calc_tot_methane(results_df)

    # weighted sum of post-storage time for calculation of weighted average. Weight is total methane potential (solid+liquid+straw)
    sum_post_storage_time = additional_data_dict['post_storage'] * methane_dict['methane_tot']

    # store methane tot and post-storage time in results df
    dfc.add_value_to_results_df(results_df, "methane_tot", "value", methane_dict['methane_tot'])
    dfc.add_value_to_results_df(results_df, "sum_post_storage_time", "value", sum_post_storage_time)

    # pre storage emissions ch4 in kg C
    carbon_emissions_dict = calc_pre_storage_emissions_carbon(methane_dict, env_config, additional_data_dict)

    """ 
    pre-storage emissions n2o and nh3
    nh3 first, then subtract nh3 emissions from n_tot for n2o emissions
    emissions in kg N
    """
    nitrogen_emissions_dict = calc_pre_storage_emissions_nitrogen(nitrogen_emissions_dict, env_config,
                                                                  additional_data_dict)

    # add emissions to results_df, note that all emissions are in CH4, NH3 or N2O!
    add_total_emissions_pre_storage(results_df, carbon_emissions_dict, nitrogen_emissions_dict)

    # Store nitrogen emissions
    add_dict_to_results(results_df, nitrogen_emissions_dict)

    # store carbon emissions dictionary in results df
    add_dict_to_results(results_df, carbon_emissions_dict)

    """calculate transport emissions"""
    fuel_consumption = transport.fuel_consumption_transport(manure_tot, additional_data_dict['distance'], env_config)
    env_impact_transport = transport.env_impact_diesel(fuel_consumption, env_config)
    dfc.add_value_to_results_df(results_df, "co2_transport", "value", env_impact_transport)

    return results_df


"""
Input list of farms from upload (contents and names), animal config and env config files
convert to pandas Dataframe
calculate manure, methane potential, pre-storage emissions and transportation costs for all farms.

"""


def calc_all_farms(list_farms, list_names, animal_config, env_config):
    results_df = dfc.create_dataframe_calc_empty()

    # do the calculations for all farms
    for farm, name in zip(list_farms, list_names):
        # get a dataframe from the uploaded farm file
        farm_df = dfc.parse_contents_to_dataframe(farm, name)
        # calculate single farm
        results_df = calc_single_farm(results_df, farm_df, animal_config, env_config)

    """calculate the weighted post-storage time, with methane potential as the weight. """
    methane_tot = dfc.find_value_farm("methane_tot", "value", results_df)
    sum_post_storage_time = dfc.find_value_farm("sum_post_storage_time", "value", results_df)
    sum_post_storage_time = sum_post_storage_time / methane_tot
    dfc.store_value_in_results_df(results_df, "sum_post_storage_time", "value", sum_post_storage_time)
    return results_df


def calc_all_farms_new(list_farms, animal_config, env_config):
    results_df = dfc.create_dataframe_calc_empty()

    # do the calculations for all farms
    for farm in list_farms:
        # calculate single farm
        results_df = calc_single_farm(results_df, farm, animal_config, env_config)

    """calculate the weighted post-storage time, with methane potential as the weight. """
    methane_tot = dfc.find_value_farm("methane_tot", "value", results_df)
    sum_post_storage_time = dfc.find_value_farm("sum_post_storage_time", "value", results_df)
    sum_post_storage_time = sum_post_storage_time / methane_tot
    dfc.store_value_in_results_df(results_df, "sum_post_storage_time", "value", sum_post_storage_time)
    return results_df


"""calculate the post storage - and field application emissions for any case"""


def post_storage_and_field_emissions(input_df, env_config, ad):
    """
    :param input_df: dataframe with input values
    :param env_config: environmental config dataframe
    :param ad: True or False, decides if the function is used to calculate emissions for manure or for digestate
    True is for digestate, False for manure
    :return: results dataframe with updated emission values for post-storage emissions and field emissions
    """

    """get the necessary values from the input dataframe"""
    c_tot, n_tot, post_storage_time = get_initial_data_field_application(input_df, ad)

    """ calc post storage emissions"""
    emissions_dict, n_tot_new = calc_post_storage_emissions(post_storage_time, c_tot, n_tot, env_config, ad)

    """calculate new c and n values for field application emissions"""
    c_tot_field = c_tot - emissions_dict['methane_emissions_storage']
    n_tot_field = n_tot_new - emissions_dict['n2o_emissions_storage']

    """calculate field application emissions"""
    if ad:
        emissions_dict['nh3_emissions_field'], emissions_dict['n2o_emissions_field'] = field.n_emissions_digestate(
            env_config, n_tot_field)

    else:
        emissions_dict['nh3_emissions_field'], emissions_dict['n2o_emissions_field'] = field.n_emissions_manure(
            env_config, n_tot_field)

    """before storing, convert all emissions from kg C/N to kg CH4/NH3/N2O"""
    emissions_dict = convert_elemental_to_molecule_weight(emissions_dict, env_config)

    """store all calculations in the input dataframe and return the dataframe"""

    store_post_storage_and_field_emissions(input_df, emissions_dict)

    """add all emissions to total emissions"""

    add_post_storage_and_field_emissions(input_df, emissions_dict)

    """make sure pre-storage emissions and transport are 0 for the no treatment pathway"""
    if not ad:
        dfc.store_value_in_results_df(input_df, "methane_emissions_pre_storage", "value", 0)
        dfc.store_value_in_results_df(input_df, "nh3_emissions_pre_storage", "value", 0)
        dfc.store_value_in_results_df(input_df, "n2o_emissions_pre_storage", "value", 0)

        dfc.store_value_in_results_df(input_df, "co2_transport", "value", 0)

    return input_df


def calc_anaerobic_digestion(input_df, env_config):
    """
    :param input_df: Input data after pre-storage and transportation (or after steam-pretreatment)
    :param env_config:
    :return: emissions and biogas yield of AD process are stored in the input_df and returned
    """
    """store data in log file for debugging"""
    logging.debug(f"input dataframe in anaerobic digestion calculations: {input_df}")
    # define new dictionary for values that are going to be stored later

    """ methane potential in m3 methane after pre-storage"""
    methane_pot = dfc.find_value_in_results_df(input_df, "methane_tot_pre_storage")

    """methane yield in m3 methane"""
    methane_yield = ad.calc_methane_yield(env_config, methane_pot)

    """ methane loss in m3 methane"""
    ad_results_dict = {'methane_emissions_ad': ad.calc_methane_loss(env_config, methane_yield)}

    """effective methane yield after loss in m3 methane"""
    ad_results_dict['methane_yield'] = methane_yield - ad_results_dict['methane_emissions_ad']

    """ methane pot left (for emissions) after ad in m3 methane"""
    effective_methane_after_ad = methane_pot - methane_yield

    """total carbon left in digestate after pre-storage and AD in kg C"""
    c_tot = total_carbon(methane_pot,
                         env_config)  # c_tot after pre-storage, calculated so that it's correct for steam-pretretment as well
    methane_yield_mass = total_carbon(methane_yield, env_config)  # methane yield in kg C
    ad_results_dict['c_tot_ad'] = c_tot - methane_yield_mass  # c_tot after ad in kg C

    """calculate biogas composition and co2 content, after losses from AD"""
    ad_results_dict['biogas'], ad_results_dict['co2_biogas'] = ad.biogas_composition(env_config,
                                                                                     ad_results_dict['methane_yield'])

    """ calculate heat and electricity demand for ad"""
    ad_results_dict['heat_demand_ad'] = ad.calc_heat_demand_ad(env_config, ad_results_dict['biogas'])
    ad_results_dict['electricity_demand_ad'] = ad.calc_electricity_demand_ad(env_config, ad_results_dict['biogas'])

    """convert methane loss from m3 to kg CH4"""
    ad_results_dict['methane_emissions_ad'] = methane_volume_to_mass(env_config,
                                                                     ad_results_dict['methane_emissions_ad'])

    """store values in input df, to then return them"""
    store_dict_in_results(input_df, ad_results_dict)

    """add up emissions"""
    dfc.add_value_to_results_df(input_df, "methane_emissions", "value", ad_results_dict['methane_emissions_ad'])
    dfc.add_value_to_results_df(input_df, "heat_demand_tot", "value", ad_results_dict['heat_demand_ad'])
    dfc.add_value_to_results_df(input_df, "electricity_demand_tot", "value", ad_results_dict['electricity_demand_ad'])

    # set methane_yield as the methane that goes to chp. In case of biogas upgrading, this value will be changed later
    dfc.store_value_in_results_df(input_df, "methane_to_chp", "value", ad_results_dict['methane_yield'])
    return input_df


def calc_chp_output(input_df, env_config):
    """
    :param input_df:
    :param env_config:
    :return:
    """
    methane_to_chp = dfc.find_value_in_results_df(input_df, "methane_to_chp")
    heat, electricity = chp.energy_produced(methane_to_chp, env_config)
    """store values in input df"""
    dfc.store_value_in_results_df(input_df, "heat_generated_chp", "value", heat)
    dfc.store_value_in_results_df(input_df, "electricity_generated_chp", "value", heat)

    """add values to total heat and electricity"""
    dfc.add_value_to_results_df(input_df, "heat_generated_tot", "value", heat)
    dfc.add_value_to_results_df(input_df, "electricity_generated_tot", "value", electricity)

    return input_df


def calc_biogas_upgrading(input_df, env_config):
    """
    :param input_df:
    :param env_config:
    :return:
    """

    # initiate dictionary for biogas upgrading results

    # get biogas value from input df
    biogas = dfc.find_value_in_results_df(input_df, "biogas")

    """calculations for biogas upgrading"""
    biogas_to_upgrading = bg.biogas_upgraded(env_config, biogas)
    biogas_to_chp = bg.biogas_chp(env_config, biogas)

    bg_results_dict = {'methane_emissions_biogas_upgrading': bg.methane_loss_upgrading(env_config,
                                                                                       bg.ch4_biogas(env_config,
                                                                                                     biogas_to_upgrading))}

    offgas_volume = bg.offgas_volume(env_config, biogas_to_upgrading)
    offgas_ch4 = bg.ch4_offgas(env_config, biogas_to_upgrading)
    offgas_co2 = bg.co2_offgas(env_config, biogas_to_upgrading)

    """subtract ch4 loss from total biomethane volume"""
    bg_results_dict['biomethane_volume'] = bg.biomethane_volume(env_config, biogas_to_upgrading) - bg_results_dict[
        'methane_emissions_biogas_upgrading']

    """subtract methane losses from the biomethane yield!"""
    bg_results_dict['biomethane_ch4'] = bg.ch4_biomethane(env_config, biogas_to_upgrading) - bg_results_dict[
        'methane_emissions_biogas_upgrading']
    biomethane_co2 = bg.co2_biomethane(env_config, biogas_to_upgrading)

    """methane volume in the biogas that goes into the CHP"""
    methane_biogas_to_chp = bg.ch4_biogas(env_config, biogas_to_chp)
    # total methane to chp, offgas plus biogas directly
    bg_results_dict['methane_to_chp'] = methane_biogas_to_chp + offgas_ch4

    """electricity demand"""
    bg_results_dict['electricity_demand_biogas_upgrading'] = bg.electricity_demand(env_config, biogas_to_upgrading)

    """electricity generated with sofc"""
    bg_results_dict['electricity_generated_sofc'] = chp.electricity_generated_sofc(env_config,
                                                                                   bg_results_dict['biomethane_ch4'])

    """convert methane loss from m3 to kg CH4"""
    bg_results_dict['methane_emissions_biogas_upgrading'] = methane_volume_to_mass(env_config, bg_results_dict['methane_emissions_biogas_upgrading'])

    """store calculations in input df"""
    store_dict_in_results(input_df, bg_results_dict)

    """add values to totals"""
    dfc.add_value_to_results_df(input_df, "electricity_demand_tot", "value",
                                bg_results_dict['electricity_demand_biogas_upgrading'])
    dfc.add_value_to_results_df(input_df, "methane_emissions", "value", bg_results_dict['methane_emissions_biogas_upgrading'])
    dfc.add_value_to_results_df(input_df, "electricity_generated_tot", "value",
                                bg_results_dict['electricity_generated_sofc'])

    """ calculate chp output as well"""
    input_df = calc_chp_output(input_df, env_config)
    return input_df


def steam_pre_treatment(input_df, env_config):
    """
    :param input_df:
    :param env_config:
    :return: updated input df with increased methane total and heat demand from steam pre treatment. Only total methane
    is updated, since the subdivision in solid, liquid and straw is not necessary anymore after that.
    """
    steam_initial_data_dict = steam_initial_data(input_df)

    """create logs for debugging"""
    logging.debug(f"initial data dict in steam pre treatment calculations: {steam_initial_data_dict}")

    """ calculate additional methane yield"""
    methane_solid_steam = steam.methane_yield_steam(env_config, steam_initial_data_dict['methane_solid_pre_storage'])
    methane_straw_steam = steam.methane_yield_steam(env_config, steam_initial_data_dict['methane_straw_pre_storage'])

    methane_tot_steam = methane_straw_steam + methane_solid_steam + steam_initial_data_dict[
        'methane_liquid_pre_storage']

    """calculate heat demand"""
    heat_demand = steam.heat_demand_steam(env_config, (
            steam_initial_data_dict['manure_straw'] + steam_initial_data_dict['manure_solid']))

    """store calculations in input df"""
    dfc.store_value_in_results_df(input_df, "methane_tot_pre_storage", "value", methane_tot_steam)
    dfc.store_value_in_results_df(input_df, "heat_demand_steam", "value", heat_demand)

    """add heat demand to total"""
    dfc.add_value_to_results_df(input_df, "heat_demand_tot", "value", heat_demand)

    return input_df


def calc_env_impacts(input_df, env_config):
    """
    :param input_df:
    :param env_config:
    :return: all environmental impacts saved in the input df and returned
    """
    """get necessary data points from input df"""
    input_dict = input_df.to_dict('records')
    logging.debug(f'input dataframe for calc_env_impacts: {input_dict}')
    initial_data_dict = get_initial_data_env_impact(input_df)

    """transport emissions are already calculated in farm calculations"""

    """calculate environmental impacts"""
    env_impact_dict = calc_gwp_impact(initial_data_dict, env_config)

    co2_eq_no_electricity = (env_impact_dict['co2_methane_tot'] + env_impact_dict['co2_n2o_tot']
                             + initial_data_dict['co2_transport'])
    env_impact_dict['co2_eq_tot'] = co2_eq_no_electricity + env_impact_dict['co2_electricity_demand_tot']

    """calculate swiss Umweltbelastungspunkte"""
    env_impact_dict['ubp_nh3'] = envi.ubp_nh3(env_config, initial_data_dict['nh3_emissions'])
    env_impact_dict['ubp_co2'] = envi.ubp_co2_eq(env_config, co2_eq_no_electricity)
    env_impact_dict['ubp_electricity_demand_non_renew'] = envi.ubp_energy_non_renew(env_config, initial_data_dict[
        'electricity_demand_tot'])
    env_impact_dict['ubp_electricity_demand_renew'] = envi.ubp_energy_renew(env_config,
                                                                            initial_data_dict['electricity_demand_tot'])

    """store impacts in input df"""
    store_dict_in_results(input_df, env_impact_dict)

    return input_df
