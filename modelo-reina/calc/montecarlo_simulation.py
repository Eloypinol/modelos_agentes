import os
import sys
import multiprocessing
import pandas as pd
import numpy as np
from dataclasses import dataclass
from calc import calcfunc
from calc.datasets import get_contacts_for_country, get_population_for_area
from calc.perf import PerfCounter
from calc.variables import set_variable, get_variable
from cythonsim import model
from tqdm import tqdm
from datetime import datetime, date, timedelta
from calc.scenarios import SCENARIOS, DefaultScenario, MitigationScenario, DefaultScenarioCV, SecondWaveScenarioCV


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
]
POP_ATTRS = [
    'susceptible', 'infected', 'detected', 'all_detected', 'hospitalized', 'in_icu',
    'dead', 'recovered', 'all_infected',
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
def get_nr_of_contacts():
    df = get_contacts_for_country()
    df = df.drop(columns='place_type').groupby('participant_age').sum()
    s = df.sum(axis=1)
    idx = list(s.index.map(lambda x: tuple([int(y) for y in x.split('-')])))
    s.index = idx
    return s.sort_index()


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
        'random_seed',
    ],
    funcs=[get_contacts_per_day, get_population_for_area],
    filedeps=[model.__file__],
)
def simulate_individuals(variables, step_callback=None):
    pc = PerfCounter()

    age_structure = get_population_for_area().values.T.flatten()

    pop_params = dict(
        age_structure=age_structure,
        contacts_per_day=get_contacts_per_day(),
    )

    hc_params = dict(hospital_beds=variables['hospital_beds'], icu_units=variables['icu_units'])
    disease_params = create_disease_params(variables)
    context = model.Context(
        population_params=pop_params,
        healthcare_params=hc_params,
        disease_params=disease_params,
        start_date=variables['start_date'],
        random_seed=variables['random_seed']
    )
    start_date = date.fromisoformat(variables['start_date'])

    for iv in variables['interventions']:
        d = (date.fromisoformat(iv[1]) - start_date).days
        if len(iv) > 2:
            val = iv[2]
        else:
            val = 0
        context.add_intervention(d, iv[0], val)
    pc.measure()

    # predicciones para una semana desde ahora
    # days = variables['simulation_days']
    first_day = datetime.strptime(variables['start_date'], '%Y-%m-%d')
    end_day = datetime.now() + timedelta(days=7)
    days = (end_day - first_day).days + 1

    extra_attrs = []
    for pop_attr in POP_ATTRS:
        for sex in ['male', 'female']:
            for i in range(0, 89, 5):
                extra_attrs.append(f'{sex}_{pop_attr}_age{i}-{i + 4}')
            extra_attrs.append(f'{sex}_{pop_attr}_age90+')

    df = pd.DataFrame(
        columns=POP_ATTRS + STATE_ATTRS + ['us_per_infected'] + extra_attrs,
        index=pd.date_range(start_date, periods=days)
    )

    for day in tqdm(range(days), leave=False, file=sys.stdout):
        s = context.generate_state()

        rec = {attr: sum(s[attr]) for attr in POP_ATTRS}

        for pop_attr in POP_ATTRS:
            for j, sex in enumerate(['male', 'female']):
                for i in range(0, 89, 5):
                    attr = f'{sex}_{pop_attr}_age{i}-{i+4}'
                    rec[attr] = sum(s[pop_attr][(i + 101 * j):(i + 101 * j + 5)])
                rec[f'{sex}_{pop_attr}_age90+'] = sum(s[pop_attr][(90 + 101 * j):(101 + 101 * j)])
            # print(f'{pop_attr}_age90+', rec[f'{pop_attr}_age90+'])

        for state_attr in STATE_ATTRS:
            rec[state_attr] = s[state_attr]

        rec['us_per_infected'] = pc.measure() * 1000 / rec['infected'] if rec['infected'] else 0
        df.sort_index()
        d = start_date + timedelta(days=day)
        df.iloc[day] = rec

        if step_callback is not None:
            ret = step_callback(df)

        context.iterate()

    return df


@calcfunc(
    variables=list(model.DISEASE_PARAMS) + [
        'sample_limit_mobility', 'max_age',
    ],
    funcs=[get_contacts_for_country],
    filedeps=[model.__file__],
)
def sample_model_parameters(what, age, severity=None, variables=None):
    max_age = variables['max_age']
    age_structure = pd.Series([1] * (max_age + 1), index=range(0, max_age + 1))
    sex_structure = pd.Series([1] * 2, index=range(0, 2))
    pop_params = dict(
        age_structure=age_structure,
        sex_structure=sex_structure,
        contacts_per_day=get_contacts_per_day(),
    )
    hc_params = dict(hospital_beds=0, icu_units=0)
    disease_params = create_disease_params(variables)
    context = model.Context(
        population_params=pop_params,
        healthcare_params=hc_params,
        disease_params=disease_params,
        start_date='2020-08-01',
    )

    if variables['sample_limit_mobility'] != 0:
        context.apply_intervention('limit-mobility', variables['sample_limit_mobility'])

    samples = context.sample(what, age, severity)

    if what == 'infectiousness':
        s = pd.Series(index=samples['day'], data=samples['val'])
        s = s[s != 0].sort_index()
        return s

    s = pd.Series(samples)
    c = s.value_counts().sort_index()
    if what == 'symptom_severity':
        c.index = c.index.map(model.SEVERITY_TO_STR)

    return c


@calcfunc(funcs=[simulate_individuals])
def simulate_monte_carlo(seed):
    set_variable('random_seed', seed)
    print(seed)
    df = simulate_individuals(skip_cache=True)
    df['run'] = seed

    return df


def run_monte_carlo(scenario_name):
    for scenario in SCENARIOS:
        if scenario.id == scenario_name:
            break
    else:
        raise Exception('Scenario not found')

    scenario.apply()

    print(scenario.id)
    with multiprocessing.Pool(processes=8) as pool:
        dfs = pool.map(simulate_monte_carlo, range(1000))

    df = pd.concat(dfs)
    df.index.name = 'date'
    df = df.reset_index()
    df['scenario'] = scenario.id
    df.to_csv('reina_%s.csv' % scenario.id, index=False)

    return df


if __name__ == '__main__':

    save_data = True

    def step_callback(df):
        rec = df.dropna().iloc[-1]

        # s = '%-12s' % rec.name.date().isoformat()
        s = '%-12s' % rec.name.isoformat()
        for attr in POP_ATTRS:
            s += '%15d' % rec[attr]

        for attr in ['exposed_per_day', 'available_hospital_beds', 'available_icu_units', 'tests_run_per_day']:
            s += '%15d' % rec[attr]
        s += '%13.2f' % rec['r']
        if rec['infected']:
            s += '%13.2f' % rec['us_per_infected']
        # print(s)
        return True


    scenario = SecondWaveScenarioCV()
    scenario.apply()
    df = simulate_individuals(step_callback=step_callback, skip_cache=True)
    # susceptible,infected,all_detected,hospitalized,in_icu,dead,recovered,all_infected,exposed_per_day
    # available_hospital_beds,available_icu_units,total_icu_units,tests_run_per_day
    # r,mobility_limitation,us_per_infected

    filename = os.path.join('..', '..', '..', 'data', 'cv_retrieved_daily.csv')
    # objectid,timestamp,positius,positius_diaris,morts,morts_diaris,recuperats,recuperats_diaris,hospitalitzats,uci,
    # casos_actius,d,e,f,g,Data,j
    df_cv = pd.read_csv(filename)
    df_cv['Data'] = pd.to_datetime(df_cv['Data'])
    df_cv = df_cv.sort_values(by=['Data'], axis=0)

    # Ajustes
    df['dead_daily'] = df['dead']
    df.dead_daily[1:] = df.dead_daily.values[1:] - df.dead.values[0:-1]
    df.infected = df.infected * 0.30
    df.detected = df.detected * 0.070

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(2, 1, figsize=(10, 12), dpi=200)
    ax[0].plot(df.index.values, df.infected, alpha=0.50, color='b', linestyle='-', label='casos activos (simulación)')
    ax[0].plot(df_cv['Data'], df_cv['casos_actius'], color='b', linestyle='--', label='casos activos (dades obertes)')

    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax[0].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
    ax[0].xaxis.grid(True)
    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax[0].grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
    ax[0].tick_params(axis='x', labelrotation=90, labelsize=6)
    ax[0].set_xlim(df.index.values[28], df.index.values[-1])

    ax[0].xaxis.grid(True)
    ax[0].yaxis.grid(False)
    ax[0].legend()

    ax[1].plot(df.index.values, df.hospitalized, alpha=0.50, color='m', linestyle='-', label='hospitalizados (simulación)')
    ax[1].plot(df.index.values, df.detected, alpha=0.50, color='g', linestyle='-', label='casos diarios (simulación)')
    ax[1].plot(df.index.values, df.dead_daily, alpha=0.50, color='k', linestyle='-', label='muertes diarias (simulación)')
    ax[1].plot(df_cv['Data'], df_cv['hospitalitzats'], color='m', linestyle='--', label='hospitalizados (dades obertes)')
    ax[1].plot(df_cv['Data'], df_cv['positius_diaris'], color='g', linestyle='--', label='casos diarios (dades obertes)')
    ax[1].plot(df_cv['Data'], df_cv['morts_diaris'], color='k', linestyle='--', label='muertes diarias (dades obertes)')

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax[1].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
    ax[1].xaxis.grid(True)
    ax[1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax[1].grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
    ax[1].tick_params(axis='x', labelrotation=90, labelsize=6)
    ax[1].set_xlim(df.index.values[28], df.index.values[-1])
    ax[1].set_ylim(0, 8000)

    ax[1].xaxis.grid(True)
    ax[1].yaxis.grid(False)
    ax[1].legend()
    timestamp = datetime.now().strftime(format='%Y-%m-%d')
    if save_data:
        plt.savefig(fname=os.path.join('..', '..', '..', 'predictions', f'{timestamp}-preds-reina.png'),
                    bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()

    df_nuria = df.loc[df.index > pd.to_datetime('01-07-2020', dayfirst=True)][['infected', 'hospitalized',
                                                                               'detected', 'dead_daily']]
    df_nuria = df_nuria.rename(columns={'infected': 'casos_activos', 'hospitalized': 'hospitalizados',
                                        'detected': 'casos_diarios', 'dead_daily': 'muertes_diarias'})
    df_nuria['casos_activos'] = df_nuria['casos_activos'].astype(dtype=int)
    df_nuria['hospitalizados'] = df_nuria['hospitalizados'].astype(dtype=int)
    df_nuria['casos_diarios'] = df_nuria['casos_diarios'].astype(dtype=int)
    df_nuria['muertes_diarias'] = df_nuria['muertes_diarias'].astype(dtype=int)
    df_nuria['fecha'] = pd.to_datetime(df_nuria.index, format='d-%m-%Y').date
    df_nuria = df_nuria[['fecha', 'casos_activos', 'hospitalizados']]
    if save_data:
        df_nuria.to_csv(os.path.join('..', '..', '..', 'predictions', f'{timestamp}-preds-reina.csv'), header=True,
                        index=False)
