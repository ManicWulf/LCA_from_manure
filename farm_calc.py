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
                       "methane_straw", "methane_tot", "ch4_emissions", "nh3_emissions", "n2o_emissions", "n_tot", "n_tot_pre_storage",
                       "transport_emissions", "sum_post_storage_time", "c_tot", "c_tot_pre_storage", "c_tot_solid", "c_tot_liquid", "c_tot_straw"]


def calc_single_farm(results_df, farm_data, animal_config, env_config):


    #total manure and methane for the farm

    methane_solid = 0
    methane_liquid = 0
    methane_straw = 0

    n_tot = 0
    manure_tot = 0

    # get a list of all animal types used from the config file (Attention: Right now it still uses animal_type_ger until Data_input.py is updated!!
    animal_types = dfc.get_animal_types(animal_config)
    for animal in animal_types:

        # get values from the farm file
        """create a log for debugging purposes"""
        logging.debug(f"Animal Type: {animal}, Value: days-outside")
        logging.debug(f"Farm Data: {farm_data.head()}")

        days_outside = dfc.find_value_farm(animal, "days-outside", farm_data)
        """debugging"""
        logging.debug(f"Retrieved days-outside for {animal}: {days_outside}")
        """"""
        hours_outside = dfc.find_value_farm(animal, "hours-outside", farm_data)
        manure_type = dfc.find_value_farm(animal, "manure-type", farm_data)
        num_animal = dfc.find_value_farm(animal, "num-animals", farm_data)

        # manure collection factor based on time spent outside of stable
        manure_collection = manure_collection_rate(hours_outside, days_outside)

        # get values from animal_config
        #check if there are any animals in the first place
        if num_animal > 0:

            # get methane potential
            methane_pot_liquid = dfc.find_value_animal_config_1_variable(animal, "methane_potential_liquid", animal_config)
            methane_pot_solid = dfc.find_value_animal_config_1_variable(animal, "methane_potential_solid", animal_config)
            methane_pot_straw = dfc.find_value_animal_config_1_variable(animal, "methane_potential_straw", animal_config)

            # get dry weight and organic dry weight for manure, and vs for straw
            dw_liquid = dfc.find_value_animal_config_1_variable(animal, "dry_weight_liquid", animal_config)
            odw_liquid = dfc.find_value_animal_config_1_variable(animal, "organic_dry_weight_liquid", animal_config)
            dw_solid = dfc.find_value_animal_config_1_variable(animal, "dry_weight_solid", animal_config)
            odw_solid = dfc.find_value_animal_config_1_variable(animal, "dry_weight_solid", animal_config)
            vs_straw = dfc.find_value_animal_config_1_variable(animal, "vs_content_straw", animal_config)

            # get Nitrogen
            n_tot_per_animal = dfc.find_value_animal_config_1_variable(animal, "nitrogen_content", animal_config)
            n_tot += n_tot_per_animal * num_animal * manure_collection


            #check if manure type exists, if it doesn't it's probably a Pig or Poultry animal with a wrong input.
            if dfc.find_value_animal_config(animal, manure_type, "animal_type", animal_config):
                #get manure total
                manure_pot_liquid = dfc.find_value_animal_config(animal, manure_type, "manure_liquid", animal_config)
                manure_pot_solid = dfc.find_value_animal_config(animal, manure_type, "manure_solid", animal_config)
                manure_pot_straw = dfc.find_value_animal_config(animal, manure_type, "manure_straw", animal_config)
                manure_solid_animal = manure_pot_solid * num_animal * manure_collection
                manure_liquid_animal = manure_pot_liquid * num_animal * manure_collection
                manure_straw_animal = manure_pot_straw * num_animal * manure_collection

                manure_tot += (manure_solid_animal + manure_straw_animal + manure_liquid_animal)

                #calculate methane potential of the farm
                methane_solid_animal = manure_solid_animal * dw_solid * odw_solid * methane_pot_solid
                methane_liquid_animal = manure_liquid_animal * dw_liquid * odw_liquid * methane_pot_liquid
                methane_straw_animal = manure_straw_animal * vs_straw * methane_pot_straw

                # add up methane potential and total manure
                dfc.add_value_to_results_df(results_df, "manure_solid", "value", manure_solid_animal)
                dfc.add_value_to_results_df(results_df, "manure_liquid", "value", manure_liquid_animal)
                dfc.add_value_to_results_df(results_df, "manure_straw", "value", manure_straw_animal)
                dfc.add_value_to_results_df(results_df, "methane_solid", "value", methane_solid_animal)
                dfc.add_value_to_results_df(results_df, "methane_liquid", "value", methane_liquid_animal)
                dfc.add_value_to_results_df(results_df, "methane_straw", "value", methane_straw_animal)



            # if the manure type doesn't exist for that animal, assume it's pig or poultry, which only have 1 manure type each and calculate with that
            else:
                #get manure total
                manure_pot_liquid = dfc.find_value_animal_config_1_variable(animal, "manure_liquid", animal_config)
                manure_pot_solid = dfc.find_value_animal_config_1_variable(animal, "manure_solid", animal_config)
                manure_pot_straw = dfc.find_value_animal_config_1_variable(animal, "manure_straw", animal_config)
                manure_solid_animal = manure_pot_solid * num_animal * manure_collection
                manure_liquid_animal = manure_pot_liquid * num_animal * manure_collection
                manure_straw_animal = manure_pot_straw * num_animal * manure_collection

                manure_tot += (manure_solid_animal + manure_straw_animal + manure_liquid_animal)

                # calculate methane potential of the farm
                methane_solid_animal = manure_solid_animal * dw_solid * odw_solid * methane_pot_solid
                methane_liquid_animal = manure_liquid_animal * dw_liquid * odw_liquid * methane_pot_liquid
                methane_straw_animal = manure_straw_animal * vs_straw * methane_pot_straw

                # add up methane potential and total manure
                dfc.add_value_to_results_df(results_df, "manure_solid", "value", manure_solid_animal)
                dfc.add_value_to_results_df(results_df, "manure_liquid", "value", manure_liquid_animal)
                dfc.add_value_to_results_df(results_df, "manure_straw", "value", manure_straw_animal)
                dfc.add_value_to_results_df(results_df, "methane_solid", "value", methane_solid_animal)
                dfc.add_value_to_results_df(results_df, "methane_liquid", "value", methane_liquid_animal)
                dfc.add_value_to_results_df(results_df, "methane_straw", "value", methane_straw_animal)

        # if the number of animals is 0, continue in the for loop to the next animal
        else:
            continue



    pre_storage = dfc.find_value_farm("pre_storage", "additional-data", farm_data)
    post_storage = dfc.find_value_farm("post_storage", "additional-data", farm_data)
    distance = dfc.find_value_farm("distance", "additional-data", farm_data)

    """get methane values from df and divide by 1000 to convert from NL to m3"""
    methane_solid = dfc.find_value_in_results_df(results_df, "methane_solid") / 1000
    methane_liquid = dfc.find_value_in_results_df(results_df, "methane_liquid") / 1000
    methane_straw = dfc.find_value_in_results_df(results_df, "methane_straw") / 1000

    methane_tot = methane_solid + methane_liquid + methane_straw

    # weighted sum of post-storage time for calculation of weighted average. Weight is total methane potential (solid+liquid+straw)
    sum_post_storage_time = post_storage * methane_tot

    # store methane tot and post-storage time in results df
    dfc.add_value_to_results_df(results_df, "methane_tot", "value", methane_tot)
    dfc.add_value_to_results_df(results_df, "sum_post_storage_time", "value", sum_post_storage_time)


    # pre storage emissions ch4 in kg C
    c_tot = total_carbon(methane_tot, env_config)
    ch4_emissions = storage.emissions_manure_ch4(pre_storage, c_tot, env_config)
    c_tot_pre_storage = c_tot - ch4_emissions
    # after subtracting the CH4-C emissions from c_tot, trun them into kg CH4
    ch4_emissions = carbon_to_methane(env_config, ch4_emissions)

    # methane pre-storage emissions by source (liquid, solid, straw)
    c_tot_solid = total_carbon(methane_solid, env_config)
    c_tot_liquid = total_carbon(methane_liquid, env_config)
    c_tot_straw = total_carbon(methane_straw, env_config)
    ch4_emissions_solid = storage.emissions_manure_ch4(pre_storage, c_tot_solid, env_config)
    ch4_emissions_liquid = storage.emissions_manure_ch4(pre_storage, c_tot_liquid, env_config)
    ch4_emissions_straw = storage.emissions_manure_ch4(pre_storage, c_tot_straw, env_config)

    """convert the emissions back to volume in m3"""
    ch4_emissions_solid_volume = carbon_to_methane_volume(ch4_emissions_solid, env_config)
    ch4_emissions_liquid_volume = carbon_to_methane_volume(ch4_emissions_liquid, env_config)
    ch4_emissions_straw_volume = carbon_to_methane_volume(ch4_emissions_straw, env_config)

    """calculate methane_pot for total, solid, liquid and straw separately after pre-storage in m3"""
    methane_solid_pre_storage = methane_solid - ch4_emissions_solid_volume
    methane_straw_pre_storage = methane_straw - ch4_emissions_straw_volume
    methane_liquid_pre_storage = methane_liquid - ch4_emissions_liquid_volume
    methane_tot_pre_storage = methane_solid_pre_storage + methane_liquid_pre_storage + methane_straw_pre_storage

    """ 
    pre-storage emissions n2o and nh3
    nh3 first, then subtract nh3 emissions from n_tot for n2o emissions
    emissions in kg N
    """
    nh3_emissions = storage.emissions_manure_nh3(pre_storage, n_tot, env_config)

    n_tot_new = n_tot - nh3_emissions
    n2o_emissions = storage.emissions_manure_n2o(pre_storage, n_tot_new, env_config)

    n_tot_pre_storage = n_tot_new - n2o_emissions

    #after adjusting n_tot_pre_storage, turn nh3 and n2o emissions into kg NH3 and kg N2O
    nh3_emissions = n_to_nh3(env_config, nh3_emissions)
    n2o_emissions = n_to_n2o(env_config, n2o_emissions)

    #add emissions to results_df, note that all emissions are in CH4-C, NH3-N or N2O-N!

    dfc.add_value_to_results_df(results_df, "ch4_emissions", "value", ch4_emissions)
    dfc.add_value_to_results_df(results_df, "nh3_emissions", "value", nh3_emissions)
    dfc.add_value_to_results_df(results_df, "n2o_emissions", "value", n2o_emissions)

    # store pre-storage emissions separately
    dfc.add_value_to_results_df(results_df, "ch4_emissions_pre_storage", "value", ch4_emissions)
    dfc.add_value_to_results_df(results_df, "nh3_emissions_pre_storage", "value", nh3_emissions)
    dfc.add_value_to_results_df(results_df, "n2o_emissions_pre_storage", "value", n2o_emissions)

    dfc.add_value_to_results_df(results_df, "c_tot_pre_storage", "value", c_tot_pre_storage)
    dfc.add_value_to_results_df(results_df, "c_tot", "value", c_tot)
    dfc.add_value_to_results_df(results_df, "n_tot", "value", n_tot)
    dfc.add_value_to_results_df(results_df, "n_tot_pre_storage", "value", n_tot_pre_storage)
    dfc.add_value_to_results_df(results_df, "methane_tot_pre_storage", "value", methane_tot_pre_storage)

    # store c_tot and methane pot seperately for liquid, manure and straw as well
    dfc.add_value_to_results_df(results_df, "c_tot_solid", "value", c_tot_solid)
    dfc.add_value_to_results_df(results_df, "c_tot_liquid", "value", c_tot_liquid)
    dfc.add_value_to_results_df(results_df, "c_tot_straw", "value", c_tot_straw)

    dfc.add_value_to_results_df(results_df, "methane_solid_pre_storage", "value", methane_solid_pre_storage)
    dfc.add_value_to_results_df(results_df, "methane_liquid_pre_storage", "value", methane_liquid_pre_storage)
    dfc.add_value_to_results_df(results_df, "methane_straw_pre_storage", "value", methane_straw_pre_storage)


    """calculate transport emissions"""
    fuel_consumption = transport.fuel_consumption_transport(manure_tot, distance, env_config)
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
    if ad:
        c_tot = dfc.find_value_in_results_df(input_df, "c_tot_ad")
    else:
        c_tot = dfc.find_value_in_results_df(input_df, "c_tot")
    n_tot = dfc.find_value_in_results_df(input_df, "n_tot")
    post_storage_time = dfc.find_value_in_results_df(input_df, "sum_post_storage_time")

    """calculate storage emissions"""
    if ad:  # check if ad is True, then calculate for digestate
        ch4_emissions_storage = storage.emissions_digestate_ch4(post_storage_time, c_tot, env_config)
        nh3_emissions_storage = storage.emissions_digestate_nh3(post_storage_time, n_tot, env_config)

    else:   # if ad is False, calculate for manure
        ch4_emissions_storage = storage.emissions_manure_ch4(post_storage_time, c_tot, env_config)
        nh3_emissions_storage = storage.emissions_manure_nh3(post_storage_time, n_tot, env_config)

    # calculate new N after nh3 emissions
    n_tot_new = n_tot - nh3_emissions_storage
    # use new value of N to calculate N2O emissions
    if ad:
        n2o_emissions_storage = storage.emissions_digestate_n2o(post_storage_time, n_tot_new, env_config)

    else:
        n2o_emissions_storage = storage.emissions_manure_n2o(post_storage_time, n_tot_new, env_config)

    """calculate new c and n values for field application emissions"""
    c_tot_field = c_tot - ch4_emissions_storage
    n_tot_field = n_tot_new - n2o_emissions_storage

    """calculate field application emissions"""
    if ad:
        nh3_emissions_field, n2o_emissions_field = field.n_emissions_digestate(env_config, n_tot_field)

    else:
        nh3_emissions_field, n2o_emissions_field = field.n_emissions_manure(env_config, n_tot_field)

    """before storing, convert all emissions from kg C/N to kg CH4/NH3/N2O"""
    nh3_emissions_storage = n_to_nh3(env_config, nh3_emissions_storage)
    n2o_emissions_storage = n_to_nh3(env_config, n2o_emissions_storage)
    ch4_emissions_storage = n_to_nh3(env_config, ch4_emissions_storage)

    nh3_emissions_field = n_to_nh3(env_config, nh3_emissions_field)
    n2o_emissions_field = n_to_nh3(env_config, n2o_emissions_field)

    """store all calculations in the input dataframe and return the dataframe"""

    dfc.store_value_in_results_df(input_df, "nh3_emissions_post_storage", "value", nh3_emissions_storage)
    dfc.store_value_in_results_df(input_df, "n2o_emissions_post_storage", "value", n2o_emissions_storage)
    dfc.store_value_in_results_df(input_df, "ch4_emissions_post_storage", "value", ch4_emissions_storage)
    dfc.store_value_in_results_df(input_df, "nh3_emissions_field", "value", nh3_emissions_field)
    dfc.store_value_in_results_df(input_df, "n2o_emissions_field", "value", n2o_emissions_field)

    """add all emissions to total emissions"""

    dfc.add_value_to_results_df(input_df, "nh3_emissions", "value", nh3_emissions_storage)
    dfc.add_value_to_results_df(input_df, "n2o_emissions", "value", n2o_emissions_storage)
    dfc.add_value_to_results_df(input_df, "ch4_emissions", "value", ch4_emissions_storage)

    dfc.add_value_to_results_df(input_df, "nh3_emissions", "value", nh3_emissions_field)
    dfc.add_value_to_results_df(input_df, "n2o_emissions", "value", n2o_emissions_field)

    """make sure pre-storage emissions and transport are 0 for the no treatment pathway"""
    if not ad:
        dfc.store_value_in_results_df(input_df, "ch4_emissions_pre_storage", "value", 0)
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

    """ methane potential in m3 methane after pre-storage"""
    methane_pot = dfc.find_value_in_results_df(input_df, "methane_tot_pre_storage")

    """store data in log file for debugging"""
    logging.debug(f"input dataframe in anaerobic digestion calculations: {input_df}")

    """methane yield in m3 methane"""
    methane_yield = ad.calc_methane_yield(env_config, methane_pot)

    """ methane loss in m3 methane"""
    methane_loss_ad = ad.calc_methane_loss(env_config, methane_yield)

    """effective methane yield after loss in m3 methane"""
    methane_yield_eff = methane_yield - methane_loss_ad

    """ methane pot left (for emissions) after ad in m3 methane"""
    effective_methane_after_ad = methane_pot - methane_yield

    """total carbon left in digestate after pre-storage and AD in kg C"""
    c_tot = total_carbon(methane_pot, env_config)     # c_tot after pre-storage, calculated so that it's correct for steam-pretretment as well
    methane_yield_mass = total_carbon(methane_yield, env_config)    # methane yield in kg C
    c_tot_ad = c_tot - methane_yield_mass       # c_tot after ad in kg C

    """calculate biogas composition and co2 content"""
    biogas, co2_content_biogas = ad.biogas_composition(env_config, methane_yield)

    """ calculate heat and electricity demand for ad"""
    heat_demand_ad = ad.calc_heat_demand_ad(env_config, biogas)
    electricity_demand_ad = ad.calc_electricity_demand_ad(env_config, biogas)

    """convert methane loss from m3 to kg CH4"""
    methane_loss_ad = methane_volume_to_mass(env_config, methane_loss_ad)

    """store values in input df, to then return them"""
    dfc.store_value_in_results_df(input_df, "methane_yield", "value", methane_yield_eff)

    # set methane_yield as the methane that goes to chp. In case of biogas upgrading, this value will be changed later
    dfc.store_value_in_results_df(input_df, "methane_to_chp", "value", methane_yield_eff)

    dfc.store_value_in_results_df(input_df, "c_tot_ad", "value", c_tot_ad)
    dfc.store_value_in_results_df(input_df, "biogas", "value", biogas)
    dfc.store_value_in_results_df(input_df, "co2_biogas", "value", co2_content_biogas)
    dfc.store_value_in_results_df(input_df, "heat_demand_ad", "value", heat_demand_ad)
    dfc.store_value_in_results_df(input_df, "electricity_demand_ad", "value", electricity_demand_ad)
    dfc.store_value_in_results_df(input_df, "methane_loss_ad", "value", methane_loss_ad)

    """add up emissions"""
    dfc.add_value_to_results_df(input_df, "ch4_emissions", "value", methane_loss_ad)
    dfc.add_value_to_results_df(input_df, "heat_demand_tot", "value", heat_demand_ad)
    dfc.add_value_to_results_df(input_df, "electricity_demand_tot", "value", electricity_demand_ad)
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
    # get biogas value from input df
    biogas = dfc.find_value_in_results_df(input_df, "biogas")

    """calculations for biogas upgrading"""
    biogas_to_upgrading = bg.biogas_upgraded(env_config, biogas)
    biogas_to_chp = bg.biogas_chp(env_config, biogas)

    ch4_loss_upgrading = bg.methane_loss_upgrading(env_config, bg.ch4_biogas(env_config, biogas_to_upgrading))

    offgas_volume = bg.offgas_volume(env_config, biogas_to_upgrading)
    offgas_ch4 = bg.ch4_offgas(env_config, biogas_to_upgrading)
    offgas_co2 = bg.co2_offgas(env_config, biogas_to_upgrading)

    """subtract ch4 loss from total biomethane volume"""
    biomethane_volume = bg.biomethane_volume(env_config, biogas_to_upgrading) - ch4_loss_upgrading

    """subtract methane losses from the biomethane yield!"""
    biomethane_ch4 = bg.ch4_biomethane(env_config, biogas_to_upgrading) - ch4_loss_upgrading
    biomethane_co2 = bg.co2_biomethane(env_config, biogas_to_upgrading)

    """methane volume in the biogas that goes into the CHP"""
    methane_biogas_to_chp = bg.ch4_biogas(env_config, biogas_to_chp)
    # total methane to chp, offgas plus biogas directly
    methane_to_chp = methane_biogas_to_chp + offgas_ch4

    """electricity demand"""
    electricity_demand = bg.electricity_demand(env_config, biogas_to_upgrading)

    """electricity generated with sofc"""
    electricity_generated = chp.electricity_generated_sofc(env_config, biomethane_ch4)

    """convert methane loss from m3 to kg CH4"""
    ch4_loss_upgrading = methane_volume_to_mass(env_config, ch4_loss_upgrading)

    """store calculations in input df"""
    dfc.store_value_in_results_df(input_df, "electricity_demand_biogas_upgrading", "value", electricity_demand)

    dfc.store_value_in_results_df(input_df, "methane_to_chp", "value", methane_to_chp)
    dfc.store_value_in_results_df(input_df, "biomethane_volume_tot", "value", biomethane_volume)
    dfc.store_value_in_results_df(input_df, "biomethane_ch4", "value", biomethane_ch4)
    dfc.store_value_in_results_df(input_df, "ch4_loss_upgrading", "value", ch4_loss_upgrading)

    dfc.store_value_in_results_df(input_df, "electricity_generated_sofc", "value", electricity_generated)

    """add values to totals"""
    dfc.add_value_to_results_df(input_df, "electricity_demand_tot", "value", electricity_demand)
    dfc.add_value_to_results_df(input_df, "ch4_emissions", "value", ch4_loss_upgrading)
    dfc.add_value_to_results_df(input_df, "electricity_generated_tot", "value", electricity_generated)

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
    manure_solid = dfc.find_value_in_results_df(input_df, "manure_solid")
    manure_straw = dfc.find_value_in_results_df(input_df, "manure_straw")
    methane_solid = dfc.find_value_in_results_df(input_df, "methane_solid_pre_storage")
    methane_straw = dfc.find_value_in_results_df(input_df, "methane_straw_pre_storage")
    methane_liquid = dfc.find_value_in_results_df(input_df, "methane_liquid_pre_storage")

    """create logs for debugging"""
    logging.debug(f"manure_solid in steam pre treatment calculations: {manure_solid}")
    logging.debug(f"manure_straw in steam pre treatment calculations: {manure_straw}")
    logging.debug(f"methane_solid in steam pre treatment calculations: {methane_solid}")
    logging.debug(f"methane_straw in steam pre treatment calculations: {methane_straw}")
    logging.debug(f"methane_liquid in steam pre treatment calculations: {methane_liquid}")

    """ calculate additional methane yield"""
    methane_solid_steam = steam.methane_yield_steam(env_config, methane_solid)
    methane_straw_steam = steam.methane_yield_steam(env_config, methane_straw)

    methane_tot_steam = methane_straw_steam + methane_solid_steam + methane_liquid

    """create logs for debugging"""
    logging.debug(f"methane_solid_steam in steam pre treatment calculations: {methane_solid_steam}")
    logging.debug(f"methane_straw_steam in steam pre treatment calculations: {methane_straw_steam}")
    logging.debug(f"methane_tot_steam in steam pre treatment calculations: {methane_tot_steam}")

    """calculate heat demand"""
    heat_demand = steam.heat_demand_steam(env_config, (manure_straw + manure_solid))

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

    methane_emissions_pre_storage = dfc.find_value_in_results_df(input_df, "ch4_emissions_pre_storage")
    methane_emissions_post_storage = dfc.find_value_in_results_df(input_df, "ch4_emissions_post_storage")
    methane_emissions_ad = dfc.find_value_in_results_df(input_df, "methane_loss_ad")
    methane_emissions_biogas_upgrading = dfc.find_value_in_results_df(input_df, "ch4_loss_upgrading")
    methane_emissions_tot = dfc.find_value_in_results_df(input_df, "ch4_emissions")

    n2o_emissions_pre_storage = dfc.find_value_in_results_df(input_df, "n2o_emissions_pre_storage")
    n2o_emissions_post_storage = dfc.find_value_in_results_df(input_df, "n2o_emissions_post_storage")
    n2o_emissions_field = dfc.find_value_in_results_df(input_df, "n2o_emissions_field")
    n2o_emissions_tot = dfc.find_value_in_results_df(input_df, "n2o_emissions")

    electricity_demand_ad = dfc.find_value_in_results_df(input_df, "electricity_demand_ad")
    electricity_demand_biogas_upgrading = dfc.find_value_in_results_df(input_df, "electricity_demand_biogas_upgrading")
    electricity_demand = dfc.find_value_in_results_df(input_df, "electricity_demand_tot")

    electricity_generated_tot = dfc.find_value_in_results_df(input_df, "electricity_generated_tot")
    heat_generated_tot = dfc.find_value_in_results_df(input_df, "heat_generated_tot")

    """transport emissions are already calculated in farm calculations"""

    nh3_tot = dfc.find_value_in_results_df(input_df, "nh3_emissions")

    """calculate environmental impacts"""
    co2_methane_pre_storage = envi.co2_methane(env_config, methane_emissions_pre_storage)
    co2_methane_post_storage = envi.co2_methane(env_config, methane_emissions_post_storage)
    co2_methane_ad = envi.co2_methane(env_config, methane_emissions_ad)
    co2_methane_biogas_upgrading = envi.co2_methane(env_config, methane_emissions_biogas_upgrading)
    co2_methane_tot = envi.co2_methane(env_config, methane_emissions_tot)

    co2_n2o_pre_storage = envi.co2_n2o(env_config, n2o_emissions_pre_storage)
    co2_n2o_post_storage = envi.co2_n2o(env_config, n2o_emissions_post_storage)
    co2_n2o_field = envi.co2_n2o(env_config, n2o_emissions_field)
    co2_n2o_tot = envi.co2_n2o(env_config, n2o_emissions_tot)

    co2_electricity_demand_ad = envi.co2_electricity_mix(env_config, electricity_demand_ad)
    co2_electricity_demand_biogas_upgrading = envi.co2_electricity_mix(env_config, electricity_demand_biogas_upgrading)
    co2_electricity_demand_tot = envi.co2_electricity_mix(env_config, electricity_demand)

    co2_eq_no_electricity = co2_methane_tot + co2_n2o_tot
    co2_eq_tot = co2_eq_no_electricity + co2_electricity_demand_tot

    co2_electricity_mix = envi.co2_electricity_mix(env_config, electricity_generated_tot)
    co2_heat_oil = envi.co2_heating_oil(env_config, heat_generated_tot)

    """calculate swiss Umweltbelastungspunkte"""
    ubp_nh3 = envi.ubp_nh3(env_config, nh3_tot)
    ubp_co2 = envi.ubp_co2_eq(env_config, co2_eq_no_electricity)
    ubp_electricity_demand_non_renew = envi.ubp_energy_non_renew(env_config, electricity_demand)
    ubp_electricity_demand_renew = envi.ubp_energy_renew(env_config, electricity_demand)

    """store impacts in input df"""
    dfc.store_value_in_results_df(input_df, "co2_methane_pre_storage", "value", co2_methane_pre_storage)
    dfc.store_value_in_results_df(input_df, "co2_methane_post_storage", "value", co2_methane_post_storage)
    dfc.store_value_in_results_df(input_df, "co2_methane_ad", "value", co2_methane_ad)
    dfc.store_value_in_results_df(input_df, "co2_methane_biogas_upgrading", "value", co2_methane_biogas_upgrading)
    dfc.store_value_in_results_df(input_df, "co2_methane_tot", "value", co2_methane_tot)

    dfc.store_value_in_results_df(input_df, "co2_n2o_pre_storage", "value", co2_n2o_pre_storage)
    dfc.store_value_in_results_df(input_df, "co2_n2o_post_storage", "value", co2_n2o_post_storage)
    dfc.store_value_in_results_df(input_df, "co2_n2o_field", "value", co2_n2o_field)
    dfc.store_value_in_results_df(input_df, "co2_n2o_tot", "value", co2_n2o_tot)

    dfc.store_value_in_results_df(input_df, "co2_electricity_demand_ad", "value", co2_electricity_demand_ad)
    dfc.store_value_in_results_df(input_df, "co2_electricity_demand_biogas_upgrading", "value", co2_electricity_demand_biogas_upgrading)
    dfc.store_value_in_results_df(input_df, "co2_electricity_demand_tot", "value", co2_electricity_demand_tot)
    dfc.store_value_in_results_df(input_df, "co2_eq_tot", "value", co2_eq_tot)

    dfc.store_value_in_results_df(input_df, "co2_electricity_mix", "value", co2_electricity_mix)
    dfc.store_value_in_results_df(input_df, "co2_heat_oil", "value", co2_heat_oil)

    dfc.store_value_in_results_df(input_df, "ubp_nh3", "value", ubp_nh3)
    dfc.store_value_in_results_df(input_df, "ubp_co2", "value", ubp_co2)
    dfc.store_value_in_results_df(input_df, "ubp_electricity_demand_non_renew", "value", ubp_electricity_demand_non_renew)
    dfc.store_value_in_results_df(input_df, "ubp_electricity_demand_renew", "value", ubp_electricity_demand_renew)

    return input_df
