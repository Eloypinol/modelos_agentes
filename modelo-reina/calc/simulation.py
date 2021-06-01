import multiprocessing
import pandas as pd
import numpy as np
from dataclasses import dataclass
from calc import calcfunc
from calc.datasets import get_contacts_for_country, get_population_for_area
from calc.perf import PerfCounter
from calc.variables import set_variable, get_variable
from cythonsim import model
from datetime import datetime, date, timedelta
from calc.scenarios import SCENARIOS, DefaultScenario, MitigationScenario, DefaultScenarioCV


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
    'susceptible', 'infected', 'all_detected', 'hospitalized', 'in_icu',
    'dead', 'recovered', 'all_infected',
]
STATE_ATTRS = [
    'exposed_per_day', 'available_hospital_beds', 'available_icu_units',
    'total_icu_units', 'tests_run_per_day', 'r', 'mobility_limitation',
]


def create_disease_params(variables):
    kwargs = {}
    for key in model.DISEASE_PARAMS:
        val = variables[key]
        if key in ('p_severe', 'p_critical', 'p_icu_death'):
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

    days = variables['simulation_days']

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

    for day in range(days):
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
        start_date='2020-01-01',
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

    df_real_data = pd.read_csv('../data/cv_retrieved_daily.csv', parse_dates=['Data'], index_col=['Data'], dayfirst=True)
    df_real_data_ucis = df_real_data['UCI']
    df_real_data_activos = df_real_data['Casos_Actius']
    df_real_data_hospitalizados = df_real_data['Hospitalitzats']
    df_real_data_infectados = df_real_data['Positius']
    df_real_data_muertos = df_real_data['Morts']

    header = '%-12s' % 'day'
    for attr in POP_ATTRS + STATE_ATTRS + ['us_per_infected']:
        header += '%15s' % attr
    # print(header)

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


    scenario = DefaultScenarioCV()
    scenario.apply()
    df = simulate_individuals(step_callback=step_callback, skip_cache=True)
    # susceptible,infected,all_detected,hospitalized,in_icu,dead,recovered,all_infected,exposed_per_day
    # available_hospital_beds,available_icu_units,total_icu_units,tests_run_per_day
    # r,mobility_limitation,us_per_infected

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns

    if False:
        attr_cols_male = []
        attr_cols_female = []
        ages = []
        attr = 'dead'
        for i in range(0, 89, 5):
            age = f'{i}-{i + 4}'
            ages.append(age)
            attr_cols_male.append(f'male_{attr}_age{age}')
            attr_cols_female.append(f'female_{attr}_age{age}')
        attr_cols_male.append(f'male_{attr}_age90+')
        attr_cols_female.append(f'female_{attr}_age90+')
        ages.append('90+')

        date = datetime.strptime('2020-05-24', '%Y-%m-%d')
        attr_male_values = -df.loc[date, attr_cols_male].values
        attr_female_values = df.loc[date, attr_cols_female].values

        attr_by_age_df = pd.DataFrame({'age': ages, 'male': attr_male_values, 'female': attr_female_values})
        ages_reversed = ages[::-1]
        palette_male = sns.color_palette("Blues")[0:7]
        palette_female = sns.color_palette("Reds")[0:7]

        fig, axes = plt.subplots(2, 1, figsize=(8, 12), dpi=200, sharex=True)
        bar_plot = sns.barplot(x='male', y='age', data=attr_by_age_df, order=ages_reversed,
                              color=palette_male[3], label='male', ax=axes[0])
        bar_plot = sns.barplot(x='female', y='age', data=attr_by_age_df, order=ages_reversed,
                              color=palette_female[3], label='female', ax=axes[0])
        bar_plot.set(xlabel=attr, ylabel='age - group', title=datetime.strftime(date, '%d/%m'))

        df_attr_real = pd.read_excel('data/IDXXX - Solicitud JC 20200520 fallecidos COVID 1.0.xlsx',
                                     sheet_name='Sheet1', header=3, parse_dates=['Fecha de alta'])
        df_attr_real = df_attr_real.pivot_table(index='Grupo edad', columns=['Sexo'], values='Éxitus', aggfunc='sum')
        df_attr_real['age'] = df_attr_real.index
        df_attr_real['Hombre'] = - df_attr_real['Hombre'].values
        df_attr_real = pd.merge(pd.DataFrame({'age': ages}), df_attr_real, on='age', how='outer').fillna(0)

        bar_plot = sns.barplot(x='Hombre', y='age', data=df_attr_real, order=ages_reversed,
                               color=palette_male[3], label='male', ax=axes[1])
        bar_plot = sns.barplot(x='Mujer', y='age', data=df_attr_real, order=ages_reversed,
                               color=palette_female[3], label='female', ax=axes[1])
        bar_plot.set(xlabel=attr, ylabel='age - group', title=datetime.strftime(date, '%d/%m'))

        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        plt.show()

        attr_cols_male = []
        attr_cols_female = []
        ages = []
        attr = 'hospitalized'
        for i in range(0, 89, 5):
            age = f'{i}-{i + 4}'
            ages.append(age)
            attr_cols_male.append(f'male_{attr}_age{age}')
            attr_cols_female.append(f'female_{attr}_age{age}')
        attr_cols_male.append(f'male_{attr}_age90+')
        attr_cols_female.append(f'female_{attr}_age90+')
        ages.append('90+')

        date = datetime.strptime('2020-05-24', '%Y-%m-%d')
        attr_male_values = -np.sum(df.loc[df.index <= date, attr_cols_male].values, axis=0)
        attr_female_values = np.sum(df.loc[df.index <= date, attr_cols_female].values, axis=0)

        attr_by_age_df = pd.DataFrame({'age': ages, 'male': attr_male_values, 'female': attr_female_values})
        ages_reversed = ages[::-1]
        palette_male = sns.color_palette("Blues")[0:7]
        palette_female = sns.color_palette("Reds")[0:7]
        fig, axes = plt.subplots(2, 1, figsize=(8, 12), dpi=200, sharex=True)
        bar_plot = sns.barplot(x='male', y='age', data=attr_by_age_df, order=ages_reversed,
                              color=palette_male[3], label='male', ax=axes[0])
        bar_plot = sns.barplot(x='female', y='age', data=attr_by_age_df, order=ages_reversed,
                              color=palette_female[3], label='female', ax=axes[0])
        bar_plot.set(xlabel=attr, ylabel='age - group', title=datetime.strftime(date, '%d/%m'))

        df_attr_real = pd.read_excel('data/IDXXX - Solicitud JC 20200520 hospitalizados COVID Y NO COVID 1.0.xlsx',
                                     sheet_name='Sheet1', header=3, parse_dates=['Fecha de hospitalización'])
        df_attr_real = df_attr_real.loc[df_attr_real['Situación COVID'] == 'COVID']
        df_attr_real = df_attr_real.pivot_table(index='Grupo de edad', columns=['Sexo'],
                                                values='Pacientes hospitalizados', aggfunc='sum')
        df_attr_real['age'] = df_attr_real.index
        df_attr_real['Hombre'] = - df_attr_real['Hombre'].values
        df_attr_real = pd.merge(pd.DataFrame({'age': ages}), df_attr_real, on='age', how='outer').fillna(0)

        bar_plot = sns.barplot(x='Hombre', y='age', data=df_attr_real, order=ages_reversed,
                               color=palette_male[3], label='male', ax=axes[1])
        bar_plot = sns.barplot(x='Mujer', y='age', data=df_attr_real, order=ages_reversed,
                               color=palette_female[3], label='female', ax=axes[1])
        bar_plot.set(xlabel=attr, ylabel='age - group', title=datetime.strftime(date, '%d/%m'))

        axes[0].set_title('simulation')
        axes[1].set_title('real')
        axes[0].legend(loc='upper left')
        axes[1].legend(loc='upper left')
        fig.tight_layout()
        plt.show()

    # exit()
    #
    df['ratio_detected'] = 1

    ratio_intervals = [
        {'start': '2020-01-01', 'duration': 0, 'from': 0.095, 'to': None},
        {'start': '2020-04-06', 'duration': 28, 'from': 0.095, 'to': 1 / 7.5},
    ]

    for interval in ratio_intervals:
        start_date = pd.Timestamp(interval['start'])
        if interval['duration'] == 0:
            # end_date = str(df.index[-1]).split(' ')[0]
            end_date = df.index[-1]
        else:
            # end_date = str(np.datetime64(start_date) + np.timedelta64(interval['duration'], 'D'))
            end_date = start_date + pd.Timedelta(interval['duration'], 'D')

        if interval['to'] is None:
            rs = interval['from']
        else:
            rs = np.arange(interval['from'], interval['to'], (interval['to'] - interval['from']) / interval['duration'])

        df.loc[(df.index > start_date) & (df.index <= end_date), 'ratio_detected'] = rs
        df.loc[(df.index > end_date), 'ratio_detected'] = interval['to']

    df['cum_inf'] = df['infected'].cumsum()
    df['percent_new_infected'] = (df['all_infected'] - df['all_infected'].shift(periods=1, fill_value=0)) / (df['all_infected'] + 1)

    detectados = 9.1     # % del total de infectados
    fallecidos = 13.8    # % de los infectados detectados acumulados
    hospitalizados = 27  # % de los infectados detectados
    uci = 21             # % de los hospitalizados
    detected_ratio = 9.1

    df['hospitalizados'] = 0.17 * df['infected'] * detected_ratio
    df['detected'] = df['infected'] * df['ratio_detected']
    df['all_detected'] = df['all_infected'] * df['ratio_detected']
    # df['hospitalizados'] = 0.23 * df['detected'] / df['ratio_detected']
    df['hospitalizados'] = 0.00003 * df['detected'] / df['ratio_detected']**4
    df['uci'] = 0.20 * df['hospitalizados']
    df['fallecidos'] = (df['all_infected'] / 80)



    # date,cases,deaths,icu,recovered,hospitalized
    df_real = pd.read_csv('../data/cv_covid19.csv', parse_dates=['date'], index_col=['date'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

    # muertes * 3
    # infected / 14
    df_real_nuria = pd.read_csv('../data/cv_real_data.csv', parse_dates=['date'], index_col=['date'])
    df_real_nuria_ucis = df_real_nuria.real_ucis
    df_real_nuria_activos = df_real_nuria.real_activos
    df_real_nuria_hospitalizados = df_real_nuria.real_hosp
    df_real_nuria_infectados = df_real_nuria.real_infected

    detection_delay = 6
    dead_delay = 12
    real_cases_nuria_dates_shifted = df_real_data_activos.index.values - np.timedelta64(detection_delay, 'D')
    model_dead_dates_shifted = df.index.values + np.timedelta64(dead_delay, 'D')

    # df_save = df[['infected', 'all_infected', 'recovered', 'detected', 'all_detected', 'ratio_detected',
    #               'hospitalizados', 'uci', 'fallecidos', 'r', 'percent_new_infected']]
    # infected_nex_day = df_save.all_infected.shift(periods=1, fill_value=0)
    # newly_infected = df_save.all_infected - infected_nex_day
    # df_save.insert(2, 'newly_infected', df_save.all_infected - infected_nex_day, True)
    # fallecidos_shifted = df_save.fallecidos.shift(periods=dead_delay, fill_value=0)
    # df_save.loc.fallecidos = df_save.fallecidos.shift(periods=dead_delay, fill_value=0)
    # df_save['date'] = df_save.index.values
    # df_save = df_save.rename(columns={'hospitalizados': 'hospitalized', 'uci': 'icu', 'fallecidos': 'dead'})

    # v = get_variable('interventions')
    # for int in v:
    #     if int[0] == 'test-with-contact-tracing':
    #         val = f'{int[2]:03d}'

    # df_save_redux = df_save[['infected', 'all_infected', 'newly_infected', 'recovered', 'detected', 'all_detected',
    #                          'hospitalized', 'icu', 'dead']]
    # df_save_redux = df_save_redux.rename(columns={
    #     'infected': 'Casos activos COVID-19',
    #     'all_infected': 'Casos acumulados COVID-19',
    #     'newly_infected': 'Nuevos casos diarios COVID-19',
    #     'recovered': 'Personas recuperadas',
    #     'detected': 'Casos detectados COVID-19',
    #     'all_detected': 'Casos detectados acumulados',
    #     'hospitalized': 'Personas hospitalizadas por COVID-19',
    #     'icu': 'Ingresados en UCI',
    #     'dead': 'Número total de fallecidos'})
    # df_save_redux.to_csv(f'WHO_model_reina_out_del03_ct{val}.csv', index=True, index_label='Fecha')

    ax.plot(df.index.values, df.susceptible, alpha=0.75, label='susceptible')
    ax.plot(df.index.values, df.detected, alpha=0.50, color='b', linestyle='-', label='infectious')
    ax.plot(df_real_data_activos.index.values, df_real_data_activos, alpha=0.50, color='b', linestyle='--', label='real infectious')
    ax.plot(real_cases_nuria_dates_shifted, df_real_data_activos, alpha=0.25, color='b', linestyle='--', label='real infectious shifted')

    ax.plot(df.index.values, df.fallecidos, alpha=0.75, color='r', linestyle='-', label='dead')
    ax.plot(model_dead_dates_shifted, df.fallecidos, alpha=0.75, color='r', linestyle='-', label='dead')
    ax.plot(df_real_data.index.values, df_real_data_muertos, color='r', linestyle='--', alpha=0.50, label='real dead')

    ax.plot(df.index.values, df.hospitalizados, alpha=0.75, color='g', linestyle='-', label='hospitalized')
    ax.plot(df_real_data_hospitalizados.index.values, df_real_data_hospitalizados, alpha=0.50, color='g', linestyle='--', label='real hospitalized')

    ax.plot(df.index.values, df.uci, alpha=0.75, color='m', linestyle='-', label='icu')
    ax.plot(df_real_data_ucis.index.values, df_real_data_ucis, alpha=0.50, color='m', linestyle='--', label='real icu')

    # real_cases = df_real.cases - df_real.recovered - df_real.deaths
    # real_cases_dates = df_real.index.values - np.timedelta64(6, 'D')
    # ax.plot(real_cases_dates, real_cases, alpha=0.25, color='b', linestyle='--', label='real infectious shifted')
    # ax.plot(df_real.index.values, real_cases, alpha=0.50, color='b', linestyle='--', label='real infectious')
    # ax.plot(df_real.index.values, df_real.deaths, color='r', linestyle='--', alpha=0.75, label='real dead')
    # ax.plot(df_real.index.values, df_real.hospitalized - df_real.deaths, color='g', linestyle='--',
    #         alpha=0.75, label='real hospitalized')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
    # ax.set_ylabel('steps')
    # ax.set_title('Activity')
    ax.xaxis.grid(True)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)
    ax.set_xlim(df.index.values[20], df.index.values[-1])

    y_lim = np.ceil(np.max(df.detected) / 2500) * 2500
    ax.set_ylim(0, y_lim)

    height = ax.get_ylim()[1]
    for intervention in get_variable('interventions')[2:]:
        if intervention[0] in ['limit-mobility', 'test-with-contact-tracing']:
            ax.axvline(datetime.strptime(intervention[1], '%Y-%m-%d'), color='k', linestyle='-', alpha=0.4)
            ax.text(datetime.strptime(intervention[1], '%Y-%m-%d'), height, f'{intervention[0]}-{intervention[2]}%',
                    {'ha': 'right', 'va': 'top'}, rotation=90, size=8, alpha=0.75)

    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend()
    plt.show()
