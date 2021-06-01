import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from calc import calcfunc
from calc.datasets import get_contacts_for_country, get_population_for_area
from calc.perf import PerfCounter
from calc.vacc_data import get_vaccination_stages_so_far
from calc.vacc_plots import plot_with_daily_cases, plot_without_daily_cases
from cythonsim import model
from tqdm import tqdm
from datetime import datetime, date, timedelta
from calc.vacc_scenarios import VaccinationScenarioCV, VaccinationSimpleScenarioCV, VaccinationScenarioWithFourthWaveCV, VaccinationScenarioCV_updated


@dataclass
class Intervention:
    name: str
    label: str
    unit: str = None


INTERVENTIONS = [
    Intervention('test-all-with-symptoms',
                 'Test all with symptoms'),
    Intervention('test-only-severe-symptoms',
                 'Test people only with severe symptoms, given percentage of mild cases are detected',
                 '%'),
    Intervention('test-with-contact-tracing',
                 'Test all with symptoms and perform contact tracing with given accuracy',
                 '%'),
    Intervention('limit-mobility',
                 'Limit population mobility',
                 '%'),
    Intervention('import-infections',
                 'Import infections',
                 'infections'),
    Intervention('build-new-hospital-beds',
                 'Build new hospital beds',
                 'beds'),
    Intervention('build-new-icu-units',
                 'Build new ICU units',
                 'units'),
    Intervention('vaccination',
                 'Vaccination',
                 'rate'),
]
POP_ATTRS = [
    'susceptible', 'infected', 'detected', 'vaccinated', 'all_detected', 'hospitalized', 'in_icu',
    'dead', 'recovered', 'all_infected', 'all_vaccinated',
]
STATE_ATTRS = [
    'exposed_per_day', 'available_hospital_beds', 'available_icu_units',
    'total_icu_units', 'tests_run_per_day', 'r', 'mobility_limitation',
]


def create_disease_params(variables):
    kwargs = {}
    start_date = date.fromisoformat(variables['start_date'])
    for key in model.DISEASE_PARAMS:
        val = variables[key]
        if key == 'p_infection':
            val = [((date.fromisoformat(d) - start_date).days, sev / 100) for d, sev in val]
        elif key in ('p_severe', 'p_critical', 'p_icu_death'):
            val = [(age, sev / 100) for age, sev in val]
        elif key.startswith('p_') or key.startswith('ratio_'):
            val = val / 100
        kwargs[key] = val

    return kwargs


@calcfunc(funcs=[get_contacts_for_country])
def get_contacts_per_day():
    df = get_contacts_for_country()
    df = pd.melt(df, id_vars=['place_type', 'participant_age'], var_name='contact_age', value_name='contacts')
    df['participant_age'] = df['participant_age'].map(lambda x: tuple([int(y) for y in x.split('-')]))
    df['contact_age'] = df['contact_age'].map(lambda x: tuple([int(y) for y in x.split('-')]))

    return df


@calcfunc(
    variables=list(model.DISEASE_PARAMS) + [
        'simulation_days', 'interventions', 'start_date',
        'hospital_beds', 'icu_units',
        'random_seed', 'vaccines', 'vaccination_stages'
    ],
    funcs=[get_contacts_per_day, get_population_for_area],
    filedeps=[model.__file__],
)
def simulate_individuals(variables, step_callback=None, delta_days=7, simulation_length=0):
    pc = PerfCounter()

    age_structure = get_population_for_area().values.T.flatten()

    pop_params = dict(
        age_structure=age_structure,
        contacts_per_day=get_contacts_per_day(),
    )

    hc_params = dict(hospital_beds=variables['hospital_beds'], icu_units=variables['icu_units'],
                     vaccines=variables['vaccines'], vaccination_stages=variables['vaccination_stages'])
    disease_params = create_disease_params(variables)
    context = model.Context(
        population_params=pop_params,
        healthcare_params=hc_params,
        disease_params=disease_params,
        start_date=variables['start_date'],
        random_seed=variables['random_seed']
    )
    start_date = date.fromisoformat(variables['start_date'])

    vacc_attrs = []
    for vacc in variables['vaccines']:
        vacc_attrs.extend([
            f"vacc_daily_{vacc['name']}_dose01",
            f"vacc_daily_{vacc['name']}_dose02",
            f"vacc_all_{vacc['name']}_dose01",
            f"vacc_all_{vacc['name']}_dose02"]
        )

    vacc_attrs.extend(['immunized', 'all_immunized'])
    for iv in variables['interventions']:
        d = (date.fromisoformat(iv[1]) - start_date).days
        if len(iv) > 2:
            val = iv[2]
        else:
            val = 0
        context.add_intervention(d, iv[0], val)
    pc.measure()

    # predicciones para 'delta_days' desde ahora
    # days = variables['simulation_days']
    first_day = datetime.strptime(variables['start_date'], '%Y-%m-%d')
    if simulation_length == 0:
        end_day = datetime.now() + timedelta(days=delta_days)
        days = (end_day - first_day).days + 1
    else:
        days = simulation_length

    extra_attrs = []
    for pop_attr in POP_ATTRS:
        for sex in ['male', 'female']:
            for i in range(0, 101, 1):
                extra_attrs.append(f'{sex}_{pop_attr}_age{i}')

    df = pd.DataFrame(
        columns=POP_ATTRS + STATE_ATTRS + vacc_attrs + ['us_per_infected'] + extra_attrs,
        index=pd.date_range(start_date, periods=days)
    )

    for day in tqdm(range(days), leave=False, file=sys.stdout):
        s = context.generate_state()

        rec = {attr: sum(s[attr]) for attr in POP_ATTRS}

        for pop_attr in POP_ATTRS:
            for j, sex in enumerate(['male', 'female']):
                for i in range(0, 101, 1):
                    attr = f'{sex}_{pop_attr}_age{i}'
                    rec[attr] = s[pop_attr][(i + 101 * j)]

        for state_attr in STATE_ATTRS:
            rec[state_attr] = s[state_attr]

        # NEW: Vaccination
        for vaccine_name in s['daily_vaccination_dose1']:
            # print(vaccine_name, s['daily_vaccination_dose1'][vaccine_name])
            rec[f'vacc_all_{vaccine_name}_dose01'] = s['all_vaccination_dose1'][vaccine_name]
            rec[f'vacc_all_{vaccine_name}_dose02'] = s['all_vaccination_dose2'][vaccine_name]
            rec[f'vacc_daily_{vaccine_name}_dose01'] = s['daily_vaccination_dose1'][vaccine_name]
            rec[f'vacc_daily_{vaccine_name}_dose02'] = s['daily_vaccination_dose2'][vaccine_name]
        rec['immunized'] = sum(s['immunized'])
        rec['all_immunized'] = sum(s['all_immunized'])

        rec['us_per_infected'] = pc.measure() * 1000 / rec['infected'] if rec['infected'] else 0
        df.sort_index()
        df.iloc[day] = rec

        if step_callback is not None:
            ret = step_callback(df)

        context.iterate()

    return df


def run_simulation(scenarios, vacc_stages, delta_days=210, save=False, debug=False):

    for scenario_name, scenario in scenarios.items():
        for vacc_plan_name, vacc_plan in vacc_stages.items():
            #vaccination_stages = get_vaccination_stages_so_far()
            vaccination_stages = []
            vaccination_stages.extend(vacc_plan)
            vacc_start_date = vacc_plan[0]['start_date']
            name = f'vacc_{scenario_name}_{vacc_plan_name}_{vacc_start_date}'
            print(name)

            scenario.apply_vaccination_stages(vaccination_stages)
            scenario.apply()
            df = simulate_individuals(step_callback=None, skip_cache=True, delta_days=delta_days)

            df['dead_daily'] = df['dead']
            df.dead_daily[1:] = df.dead_daily.values[1:] - df.dead.values[0:-1]

            # Adjust infected
            df.infected = (df.infected * 0.30).astype(int)
            idx1 = df.columns.tolist().index('male_infected_age0')
            idx2 = df.columns.tolist().index('female_infected_age100')
            for i in range(idx1, idx2 + 1):
                df.iloc[:, i] = df.iloc[:, i] * 0.30

            # Adjust detected
            df.detected = (df.detected * 0.060).astype(int)
            idx1 = df.columns.tolist().index('male_detected_age0')
            idx2 = df.columns.tolist().index('female_detected_age100')
            for i in range(idx1, idx2 + 1):
                df.iloc[:, i] = df.iloc[:, i] * 0.060

            # Round to int
            idx_r = df.columns.tolist().index('r')
            for i in range(df.shape[1]):
                if i != idx_r:
                    df.iloc[:, i] = df.iloc[:, i].apply(np.floor).astype(int)

            # plot_with_daily_cases(df, debug=DEBUG)
            plt = plot_with_daily_cases(df, debug=debug) # ALC with
            plt.tight_layout()

            keep_columns = ['infected', 'dead_daily', 'vaccinated', 'all_vaccinated', 'immunized', 'all_immunized']
            keep_columns.extend([name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose01')])
            keep_columns.extend([name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose02')])
            keep_columns.extend([name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose01')])
            keep_columns.extend([name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose02')])

            df_vacc = df.loc[df.index > pd.to_datetime('01-07-2020', dayfirst=True)]
            df_vacc = df_vacc[keep_columns]
            df_vacc = df_vacc.rename(columns={'infected': 'casos_activos', 'dead_daily': 'muertes_diarias',
                                              'vaccination_min_age': 'edad_minima_vacunacion',
                                              'vaccination_max_age': 'edad_maxima_vacunacion',
                                              'vaccinated': 'vacunados_diarios', 'all_vaccinated': 'vacunados_totales',
                                              'immunized': 'inmunizados_diarios', 'all_immunized': 'inmunizados_totales',
                                              })
            df_vacc['fecha'] = pd.to_datetime(df_vacc.index, format='d-%m-%Y').date

            if save:
                filename = os.path.join('predictions', name)
                plt.savefig(f'{filename}.png')
                df_vacc.to_csv(f'{filename}.csv', header=True, index=False)
            #plt.show()

    return


if __name__ == '__main__':

    start_date = '2020-06-01'
    scenarios = {
        #'4thwave': VaccinationScenarioWithFourthWaveCV(start_date=start_date),
        #'3rdwave': VaccinationScenarioCV(start_date=start_date),
        'Updated': VaccinationScenarioCV_updated(start_date=start_date)
    }

    vacc_start_date = '2021-02-04'
    vacc_stages = {
        'without_vaccination': [
            {'start_date': vacc_start_date, 'end_date': '2021-03-15',
             'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 70, 'max_age': 100, 'rate': 0},
        ],
        'with_vaccination': get_vaccination_stages_so_far(),
        #],
        
        #'with_vaccination': [
            # febrero -> 200.000
            #{'start_date': vacc_start_date, 'end_date': '2021-02-28',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 80, 'max_age': 100, 'rate': 3000},
            #{'start_date': vacc_start_date, 'end_date': '2021-02-28',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 80, 'max_age': 100, 'rate': 4000},

            # marzo  -> 270.000
            #{'start_date': '2021-03-01', 'end_date': '2021-03-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 70, 'max_age': 100, 'rate': 4000},
            #{'start_date': '2021-03-01', 'end_date': '2021-03-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 70, 'max_age': 100, 'rate': 4700},

            # abril  -> 300.000
            #{'start_date': '2021-04-01', 'end_date': '2021-04-30',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 60, 'max_age': 100, 'rate': 4500},
            #{'start_date': '2021-04-01', 'end_date': '2021-04-30',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 60, 'max_age': 100, 'rate': 5500},

            # mayo   -> 350.000
            #{'start_date': '2021-05-01', 'end_date': '2021-05-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 60, 'max_age': 80, 'rate': 5000},
            #{'start_date': '2021-05-01', 'end_date': '2021-05-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 60, 'max_age': 80, 'rate': 6000},

            # junio  -> 400.000
            #{'start_date': '2021-06-01', 'end_date': '2021-06-30',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 60, 'max_age': 80, 'rate': 6000},
            #{'start_date': '2021-06-01', 'end_date': '2021-06-30',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 60, 'max_age': 80, 'rate': 7000},

            # julio  -> 400.000
            #{'start_date': '2021-07-01', 'end_date': '2021-07-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 50, 'max_age': 60, 'rate': 6000},
            #{'start_date': '2021-07-01', 'end_date': '2021-07-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 50, 'max_age': 60, 'rate': 7000},

            # agosto  -> 400.000
            #{'start_date': '2021-08-01', 'end_date': '2021-08-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 1, 'min-age': 50, 'max_age': 60, 'rate': 6000},
            #{'start_date': '2021-08-01', 'end_date': '2021-08-31',
            # 'vaccine': 'Pfizer-BioNTech', 'dose': 2, 'min-age': 50, 'max_age': 60, 'rate': 7000},
        #]
    }

    run_simulation(scenarios, vacc_stages, delta_days=210, save=True, debug=False)
