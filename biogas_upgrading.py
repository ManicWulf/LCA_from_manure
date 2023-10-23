"""
Contains all functions regarding biogas upgrading,
Biogas yield, biomethane yield, energy costs etc.

"""
import dash_functions_and_callbacks as dfc


def get_biogas_composition(env_config):
    ch4_content = dfc.find_value_env_config("methane_biogas", "value", env_config)
    co2_content = dfc.find_value_env_config("co2_biogas", "value", env_config)
    return ch4_content, co2_content


def get_upgrade_percentage(env_config):
    return dfc.find_value_env_config("upgrade_factor_biogas", "value", env_config)


def get_chp_percentage(env_config):
    return dfc.find_value_env_config("chp_flow_percentage_biogas_upgrading", "value", env_config)


def get_electricity_demand(env_config):
    return dfc.find_value_env_config("electricity_demand_upgrading", "value", env_config)


def get_methane_loss(env_config):
    return dfc.find_value_env_config("methane_loss_upgrading", "value", env_config)


def get_offgas_ch4_content(env_config):
    return dfc.find_value_env_config("methane_content_offgas_upgrading", "value", env_config)


def get_offgas_co2_content(env_config):
    return dfc.find_value_env_config("co2_content_offgas_upgrading", "value", env_config)


def get_co2_content_upgraded(env_config):
    return dfc.find_value_env_config("co2_content_upgraded", "value", env_config)


def get_methane_content_upgraded(env_config):
    return dfc.find_value_env_config("methane_content_upgraded", "value", env_config)


def get_biomethane_ch4_to_co2_ratio(env_config):
    return dfc.find_value_env_config("biomethane_ch4_to_co2_ratio", "value", env_config)





"""functions for calculations"""

def biogas_upgraded(env_config, biogas):
    return get_upgrade_percentage(env_config) * biogas


def biogas_chp(env_config, biogas):
    return get_chp_percentage(env_config) * biogas


def methane_loss_upgrading(env_config, methane):
    return methane * get_methane_loss(env_config)


def offgas_volume(env_config, biogas):
    biogas_ch4, biogas_co2 = get_biogas_composition(env_config)
    ch4_loss = get_methane_loss(env_config)
    biomethane_ch4_co2_ratio = get_biomethane_ch4_to_co2_ratio(env_config)
    offgas_ch4 = get_offgas_ch4_content(env_config)
    offgas_co2 = get_offgas_co2_content(env_config)
    return biogas * ((biogas_ch4 - biogas_ch4 * ch4_loss - biomethane_ch4_co2_ratio * biogas_co2) / (offgas_ch4 - biomethane_ch4_co2_ratio * offgas_co2))


def biomethane_volume(env_config, biogas):
    offgas = offgas_volume(env_config, biogas)
    biogas_ch4, biogas_co2 = get_biogas_composition(env_config)
    offgas_co2 = get_offgas_co2_content(env_config)
    biomethane_co2 = get_co2_content_upgraded(env_config)
    return (biogas_co2 * biogas - offgas_co2 * offgas) / biomethane_co2


def co2_offgas(env_config, biogas):
    return get_offgas_co2_content(env_config) * offgas_volume(env_config, biogas)


def ch4_offgas(env_config, biogas):
    return get_offgas_ch4_content(env_config) * offgas_volume(env_config, biogas)


def co2_biomethane(env_config, biogas):
    return get_co2_content_upgraded(env_config) * biomethane_volume(env_config, biogas)


def ch4_biomethane(env_config, biogas):
    return get_methane_content_upgraded(env_config) * biomethane_volume(env_config, biogas)


def ch4_biogas(env_config, biogas):
    ch4_content, co2_content = get_biogas_composition(env_config)
    return ch4_content * biogas


def co2_biogas(env_config, biogas):
    ch4_content, co2_content = get_biogas_composition(env_config)
    return co2_content * biogas


def electricity_demand(env_config, biogas):
    return biogas * get_electricity_demand(env_config)







