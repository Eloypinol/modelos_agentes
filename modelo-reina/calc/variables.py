import json
import hashlib
from contextlib import contextmanager


# Variables
VARIABLE_DEFAULTS = {
    'area_name': 'CV',
    'area_name_long': 'Comunitat Valenciana',
    'country': 'IT',
    'max_age': 100,
    'simulation_days': 142,
    'start_date': '2020-08-01',
    'hospital_beds': 18992,
    'icu_units': 965,

    #
    # Disease parameters
    #

    # Chance to be asymptomatic
    'p_asymptomatic': 50.0,  # %

    # Overall chance to become infected after being exposed.
    # This is modified by viral load of the infector, which
    # depends on the day of the illness.
    # 'p_infection': 30.0,  # %

    # NEW: consider new strains
    'p_infection': [
        ('2020-01-01', 30.0),
        ('2020-12-01', 32),
        
    ],

    # Chance to die after regular hospital care
    'p_hospital_death': 0.0,  # %
    # Chance to die after ICU care
    'p_icu_death': [
        [0, 40.0],
        [10, 40.0],
        [20, 50.0],
        [30, 50.0],
        [40, 50.0],
        [50, 50.0],
        [60, 60.0],
        [70, 68.50],
        [80, 80.50]
    ],

    # Chance to die if no hospital beds are available (but not needing ICU care)
    'p_hospital_death_no_beds': 20.0,  # %
    # Chance to die if no ICU care units are available
    'p_icu_death_no_beds': 100.0,  # %

    'mean_incubation_duration': 5.1,
    'mean_duration_from_onset_to_death': 18.8,
    'mean_duration_from_onset_to_recovery': 21.0,

    'ratio_of_duration_before_hospitalisation': 30.0,  # %
    'ratio_of_duration_in_ward': 15.0,  # %

    # Ratio of all symptomatic people that require hospitalization
    # (more than mild symptoms) by age group
    # Numbers scaled, because source assumes 50% asymptomatic people.
    # Source: https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf
    'p_severe': [
        [0, 0.0],
        [10, 0.0816],
        [20, 2.08],
        [30, 6.86],
        [40, 8.5],
        [50, 16.32],
        [60, 45.0],
        [70, 87.0],
        [80, 95.0],
        [90, 98.0],
    ],
    # 'p_severe': [
    #     [0, 0.0],
    #     [10, 0.0816],
    #     [20, 1.08],
    #     [30, 3.86],
    #     [40, 4.5],
    #     [55, 16.32],
    #     [60, 43.6],
    #     [70, 92.2],
    #     [80, 95.0],
    #     [90, 98.0],
    # ],
    # Ratio of hospitalized cases requiring critical (ICU) care
    # Source: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf
    'p_critical': [
        [0, 5.0],
        [10, 5.0],
        [20, 5.0],
        [30, 5.0],
        [40, 6.3],
        [50, 12.2],
        [60, 27.4],
        [70, 43.2],
        [80, 70.9]
    ],

    # 'p_critical': [
    #     [0, 5.0],
    #     [10, 5.0],
    #     [20, 5.0],
    #     [30, 5.0],
    #     [40, 6.3],
    #     [50, 12.2],
    #     [60, 27.4],
    #     [70, 43.2],
    #     [80, 70.9]
    # ],

    'interventions': [],

    'preset_scenario': 'default',

    # Used for sampling the model
    'sample_limit_mobility': 1,
    # Used for Monte Carlo simulation
    'random_seed': 42,

    'vaccines': [],

    'vaccination_stages': []
}


session = {}


def set_variable(var_name, value):
    assert var_name in VARIABLE_DEFAULTS
    assert isinstance(value, type(VARIABLE_DEFAULTS[var_name]))

    if value == VARIABLE_DEFAULTS[var_name]:
        if var_name in session:
            del session[var_name]
        return

    session[var_name] = value


def get_variable(var_name, var_store=None):
    out = None

    if var_store is not None:
        out = var_store.get(var_name)
    elif var_name in session:
        out = session[var_name]

    if out is None:
        out = VARIABLE_DEFAULTS[var_name]

    if isinstance(out, list):
        # Make a copy
        return list(out)

    return out


def reset_variable(var_name):
    if var_name in session:
        del session[var_name]


def reset_variables():
    session.clear()
