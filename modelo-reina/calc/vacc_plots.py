import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta


def plot_with_daily_cases(df, debug=False):

    vaccines_daily_dose01 = [name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose01')]
    vaccines_all_dose01 = [name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose01')]
    vaccines_daily_dose02 = [name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose02')]
    vaccines_all_dose02 = [name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose02')]

    filename = os.path.join('data', 'cv_retrieved_daily.csv')
    df_cv = pd.read_csv(filename)
    df_cv['Data'] = pd.to_datetime(df_cv['Data'])
    df_cv = df_cv.sort_values(by=['Data'], axis=0)

    fig, ax = plt.subplots(4, 2, figsize=(25, 15), dpi=200)
    ax[0, 0].plot(df.index.values, df.infected, alpha=0.75, color='b', linestyle='-', label='simulación')
    ax[0, 1].plot(df.index.values, df.dead_daily, alpha=0.75, color='k', linestyle='-', label='simulación')
    ax[1, 0].plot(df.index.values, df.detected, alpha=0.75, color='g', linestyle='-', label='simulación')
    ax[1, 1].plot(df.index.values, df.hospitalized, alpha=0.75, color='m', linestyle='-', label='simulación')

    for v in vaccines_daily_dose01:
        vacc_name = v[11:-7]
        ax[2, 0].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', label=f'{vacc_name} dosis 1')

    for v in vaccines_daily_dose02:
        vacc_name = v[11:-7]
        ax[2, 0].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', linewidth=3, label=f'{vacc_name} dosis 2')

    for v in vaccines_all_dose01:
        vacc_name = v[9:-7]
        ax[2, 1].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', label=f'{vacc_name} dosis 1')

    for v in vaccines_all_dose02:
        vacc_name = v[9:-7]
        ax[2, 1].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', linewidth=3, label=f'{vacc_name} dosis 2')

    ax[3, 0].plot(df.index.values, df.immunized, alpha=0.75, color='b', linestyle='-', label='simulación')
    ax[3, 1].plot(df.index.values, df.all_immunized, alpha=0.75, color='b', linestyle='-', label='simulación')

    if debug:
        ax[0, 0].set_ylim(0, 2000)
        ax[0, 1].set_ylim(0, 20)
        ax[1, 1].set_ylim(0, 500)
        ax[1, 0].set_ylim(0, 50)
        ax[2, 0].set_ylim(0, 200)
        ax[2, 1].set_ylim(0, 5000)
        ax[3, 0].set_ylim(0, 250)
        ax[3, 1].set_ylim(0, 2000)
    else:
        ax[0, 0].plot(df_cv['Data'], df_cv['casos_actius'], alpha=0.50, color='b', linestyle='--', label='dades obertes')
        ax[1, 0].plot(df_cv['Data'], df_cv['positius_diaris'], alpha=0.50, color='g', linestyle='--', label='dades obertes')
        ax[1, 1].plot(df_cv['Data'], df_cv['hospitalitzats'], alpha=0.50, color='m', linestyle='--', label='dades obertes')
        ax[0, 1].plot(df_cv['Data'], df_cv['morts_diaris'], alpha=0.50, color='k', linestyle='--', label='dades obertes')
        ax[0, 0].set_ylim(0, 80000)
        ax[0, 1].set_ylim(0, 120)
        ax[1, 0].set_ylim(0, 10000)
        ax[1, 1].set_ylim(0, 7000)
        ax[2, 0].set_ylim(0, 40000)
        ax[2, 1].set_ylim(0, 4500000)
        ax[3, 0].set_ylim(0, 40000)
        ax[3, 1].set_ylim(0, 4500000)

    ax[0, 0].set_title('Casos activos')
    ax[0, 1].set_title('Muertes diarias')
    ax[1, 0].set_title('Casos diarios')
    ax[1, 1].set_title('Hospitalizados')
    ax[2, 0].set_title('Vacunaciones diarias')
    ax[2, 1].set_title('Vacunaciones totales')
    ax[3, 0].set_title('Inmunizados diarios')
    ax[3, 1].set_title('Inmunizados totales')

    for i in range(4):
        for j in range(2):
            ax[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax[i, j].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
            ax[i, j].xaxis.grid(True)
            ax[i, j].grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
            ax[i, j].tick_params(axis='x', labelrotation=90, labelsize=6)
            if debug:
                ax[i, j].xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                ax[i, j].set_xlim(df.index.values[28], df.index.values[-1])
                ax[i, j].xaxis.set_major_locator(mdates.DayLocator(interval=7))

            ax[i, j].xaxis.grid(True)
            ax[i, j].yaxis.grid(False)
            ax[i, j].legend()

    return plt


def plot_with_daily_cases_updated(df, debug=False):

    vaccines_daily_dose01 = [name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose01')]
    vaccines_all_dose01 = [name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose01')]
    vaccines_daily_dose02 = [name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose02')]
    vaccines_all_dose02 = [name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose02')]

    filename = os.path.join('data', 'cv_retrieved_daily.csv')
    df_cv = pd.read_csv(filename)
    df_cv['Data'] = pd.to_datetime(df_cv['Data'])
    df_cv = df_cv.sort_values(by=['Data'], axis=0)

    fig, ax = plt.subplots(4, 1, figsize=(15, 20), dpi=200)
    ax[0].plot(df.index.values, df.infected, alpha=0.75, color='b', linestyle='-', label='simulación')
    ax[1].plot(df.index.values, df.dead_daily, alpha=0.75, color='k', linestyle='-', label='simulación')

    for v in vaccines_daily_dose01:
        vacc_name = v[11:-7]
        ax[2].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', label=f'{vacc_name} dosis 1')

    for v in vaccines_daily_dose02:
        vacc_name = v[11:-7]
        ax[2].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', linewidth=3, label=f'{vacc_name} dosis 2')

    ax[3].plot(df.index.values, df.all_immunized, alpha=0.75, color='b', linestyle='-', label='simulación')

    if debug:
        ax[0].set_ylim(0, 2000)
        ax[1].set_ylim(0, 20)
        ax[2].set_ylim(0, 200)
        ax[3].set_ylim(0, 2000)
    else:
        ax[0].plot(df_cv['Data'], df_cv['casos_actius'], alpha=0.50, color='b', linestyle='--', label='dades obertes')
        ax[1].plot(df_cv['Data'], df_cv['morts_diaris'], alpha=0.50, color='k', linestyle='--', label='dades obertes')
        ax[0].set_ylim(0, 80000)
        ax[1].set_ylim(0, 120)
        ax[2].set_ylim(0, 50000)
        ax[3].set_ylim(0, 4500000)

    ax[0].set_title('Casos activos')
    ax[1].set_title('Muertes diarias')
    ax[2].set_title('Vacunaciones diarias')
    ax[3].set_title('Inmunizados totales')

    for i in range(4):
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax[i].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
        ax[i].xaxis.grid(True)
        ax[i].grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
        ax[i].tick_params(axis='x', labelrotation=90, labelsize=6)
        if debug:
            ax[i].xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            ax[i].set_xlim(df.index.values[28], df.index.values[-1])
            ax[i].xaxis.set_major_locator(mdates.DayLocator(interval=7))

        ax[i].xaxis.grid(True)
        ax[i].yaxis.grid(False)
        ax[i].legend()

    return plt


def plot_without_daily_cases(df, debug=False):

    vaccines_daily_dose01 = [name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose01')]
    vaccines_all_dose01 = [name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose01')]
    vaccines_daily_dose02 = [name for name in df.columns if name.startswith('vacc_daily') and name.endswith('dose02')]
    vaccines_all_dose02 = [name for name in df.columns if name.startswith('vacc_all') and name.endswith('dose02')]

    filename = os.path.join('data', 'cv_retrieved_daily.csv')
    df_cv = pd.read_csv(filename)
    df_cv['Data'] = pd.to_datetime(df_cv['Data'])
    df_cv = df_cv.sort_values(by=['Data'], axis=0)

    fig, ax = plt.subplots(3, 2, figsize=(21, 15), dpi=200)
    ax[0, 0].plot(df.index.values, df.infected, alpha=0.75, color='b', linestyle='-', label='simulación')
    ax[0, 1].plot(df.index.values, df.dead_daily, alpha=0.75, color='k', linestyle='-', label='simulación')

    for v in vaccines_daily_dose01:
        vacc_name = v[11:-7]
        ax[1, 0].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', label=f'{vacc_name} dosis 1')

    for v in vaccines_daily_dose02:
        vacc_name = v[11:-7]
        ax[1, 0].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', linewidth=3, label=f'{vacc_name} dosis 2')

    for v in vaccines_all_dose01:
        vacc_name = v[9:-7]
        ax[1, 1].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', label=f'{vacc_name} dosis 1')

    for v in vaccines_all_dose02:
        vacc_name = v[9:-7]
        ax[1, 1].plot(df.index.values, df[v], alpha=0.75, color='b', linestyle='-', linewidth=3, label=f'{vacc_name} dosis 2')

    ax[2, 0].plot(df.index.values, df.immunized, alpha=0.75, color='b', linestyle='-', label='simulación')
    ax[2, 1].plot(df.index.values, df.all_immunized, alpha=0.75, color='b', linestyle='-', label='simulación')

    if debug:
        ax[0, 0].set_ylim(0, 2000)
        ax[0, 1].set_ylim(0, 20)
        ax[1, 0].set_ylim(0, 200)
        ax[1, 1].set_ylim(0, 5000)
        ax[2, 0].set_ylim(0, 250)
        ax[2, 1].set_ylim(0, 2000)
    else:
        ax[0, 0].plot(df_cv['Data'], df_cv['casos_actius'], alpha=0.50, color='b', linestyle='--', label='dades obertes')
        ax[0, 0].set_ylim(0, 60000)
        ax[0, 1].set_ylim(0, 150)
        ax[1, 0].set_ylim(0, 50000)
        ax[1, 1].set_ylim(0, 4500000)
        ax[2, 0].set_ylim(0, 50000)
        ax[2, 1].set_ylim(0, 4500000)

    ax[0, 0].set_title('Casos activos')
    ax[0, 1].set_title('Muertes diarias')
    ax[1, 0].set_title('Vacunaciones diarias')
    ax[1, 1].set_title('Vacunaciones totales')
    ax[2, 0].set_title('Inmunizados diarios')
    ax[2, 1].set_title('Inmunizados totales')

    for i in range(3):
        for j in range(2):
            ax[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax[i, j].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
            ax[i, j].xaxis.grid(True)
            ax[i, j].grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
            ax[i, j].tick_params(axis='x', labelrotation=90, labelsize=6)
            if debug:
                ax[i, j].xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                ax[i, j].set_xlim(df.index.values[28], df.index.values[-1])
                ax[i, j].xaxis.set_major_locator(mdates.DayLocator(interval=7))

            ax[i, j].xaxis.grid(True)
            ax[i, j].yaxis.grid(False)
            ax[i, j].legend()

    return plt


def plot_vacc_comparison(df1, df2, mark_date):

    vaccines_daily_dose01_1 = [name for name in df1.columns if name.startswith('vacc_daily') and name.endswith('dose01')]
    vaccines_all_dose01_1 = [name for name in df1.columns if name.startswith('vacc_all') and name.endswith('dose01')]
    vaccines_daily_dose02_1 = [name for name in df1.columns if name.startswith('vacc_daily') and name.endswith('dose02')]
    vaccines_all_dose02_1 = [name for name in df1.columns if name.startswith('vacc_all') and name.endswith('dose02')]

    vaccines_daily_dose01_2 = [name for name in df2.columns if name.startswith('vacc_daily') and name.endswith('dose01')]
    vaccines_all_dose01_2 = [name for name in df2.columns if name.startswith('vacc_all') and name.endswith('dose01')]
    vaccines_daily_dose02_2 = [name for name in df2.columns if name.startswith('vacc_daily') and name.endswith('dose02')]
    vaccines_all_dose02_2 = [name for name in df2.columns if name.startswith('vacc_all') and name.endswith('dose02')]

    filename = os.path.join('data', 'cv_retrieved_daily.csv')
    df_cv = pd.read_csv(filename)
    df_cv['Data'] = pd.to_datetime(df_cv['Data'])
    df_cv = df_cv.sort_values(by=['Data'], axis=0)

    start_date1 = date.fromisoformat(df1.fecha.iloc[0])
    end_date = date.fromisoformat(df1.fecha.iloc[-1])
    days = (end_date - start_date1).days + 1

    df1.index = pd.date_range(start_date1, periods=days)
    df2.index = pd.date_range(start_date1, periods=days)
    df2 = df2.loc[df2.index >= pd.to_datetime(mark_date)]

    d = datetime.strptime(mark_date, '%Y-%M-%d').strftime('%d-%M-%Y')

    colormap = plt.get_cmap('tab10')
    colors = [colormap(0), colormap(2), colormap(1)]
    fig, ax = plt.subplots(3, 2, figsize=(21, 15), dpi=200)
    ax[0, 0].plot(df1.index.values, df1.casos_activos, alpha=0.75, color=colors[0], linestyle='-', label=f'simulación sin vacunación a partir del {d}')
    ax[0, 1].plot(df1.index.values, df1.muertes_diarias, alpha=0.75, color=colors[0], linestyle='-', label=f'simulación sin vacunación a partir del {d}')

    ax[0, 0].plot(df2.index.values, df2.casos_activos, alpha=0.75, color=colors[1], linestyle='-', label=f'simulación con vacunación a partir del {d}')
    ax[0, 1].plot(df2.index.values, df2.muertes_diarias, alpha=0.75, color=colors[1], linestyle='-', label=f'simulación con vacunación a partir del {d}')

    for v in vaccines_daily_dose01_1:
        vacc_name = v[11:-7]
        ax[1, 0].plot(df1.index.values, df1[v], alpha=0.75, color=colors[0], linestyle='-', label=f'1ª dosis (antes del {d})')

    for v in vaccines_daily_dose02_1:
        vacc_name = v[11:-7]
        ax[1, 0].plot(df1.index.values, df1[v], alpha=0.75, color=colors[0], linestyle='-', linewidth=3, label=f'2ª dosis (antes del {d})')

    for v in vaccines_all_dose01_1:
        vacc_name = v[9:-7]
        ax[1, 1].plot(df1.index.values, df1[v], alpha=0.75, color=colors[0], linestyle='-', label=f'1ª dosis (antes del {d})')

    for v in vaccines_all_dose02_1:
        vacc_name = v[9:-7]
        ax[1, 1].plot(df1.index.values, df1[v], alpha=0.75, color=colors[0], linestyle='-', linewidth=3, label=f'2ª dosis (antes del {d})')

    for v in vaccines_daily_dose01_2:
        vacc_name = v[11:-7]
        ax[1, 0].plot(df2.index.values, df2[v], alpha=0.75, color=colors[1], linestyle='-', label=f'1ª dosis (después del {d})')

    for v in vaccines_daily_dose02_2:
        vacc_name = v[11:-7]
        ax[1, 0].plot(df2.index.values, df2[v], alpha=0.75, color=colors[1], linestyle='-', linewidth=3, label=f'2ª dosis (después del {d})')

    for v in vaccines_all_dose01_2:
        vacc_name = v[9:-7]
        ax[1, 1].plot(df2.index.values, df2[v], alpha=0.75, color=colors[1], linestyle='-', label=f'1ª dosis (después del {d})')

    for v in vaccines_all_dose02_2:
        vacc_name = v[9:-7]
        ax[1, 1].plot(df2.index.values, df2[v], alpha=0.75, color=colors[1], linestyle='-', linewidth=3, label=f'2ª dosis (después del {d})')

    ax[2, 0].plot(df1.index.values, df1.inmunizados_diarios, alpha=0.75, color=colors[0], linestyle='-', label=f'antes del {d}')
    ax[2, 0].plot(df2.index.values, df2.inmunizados_diarios, alpha=0.75, color=colors[1], linestyle='-', label=f'después del {d}')
    ax[2, 1].plot(df1.index.values, df1.inmunizados_totales, alpha=0.75, color=colors[0], linestyle='-', label=f'antes del {d}')
    ax[2, 1].plot(df2.index.values, df2.inmunizados_totales, alpha=0.75, color=colors[1], linestyle='-', label=f'después del {d}')

    ax[0, 0].plot(df_cv['Data'], df_cv['casos_actius'], alpha=0.25, color='b', linestyle='--', label='casos reales')
    ax[0, 0].set_ylim(0, 60000)
    ax[0, 1].set_ylim(0, 150)
    ax[1, 0].set_ylim(0, 40000)
    ax[1, 1].set_ylim(0, 4500000)
    ax[2, 0].set_ylim(0, 40000)
    ax[2, 1].set_ylim(0, 4500000)

    ax[0, 0].set_title('Casos activos')
    ax[0, 1].set_title('Muertes diarias')
    ax[1, 0].set_title('Vacunaciones diarias')
    ax[1, 1].set_title('Vacunaciones totales')
    ax[2, 0].set_title('Inmunizados diarios')
    ax[2, 1].set_title('Inmunizados totales')

    for i in range(3):
        for j in range(2):
            ax[i, j].axvline(pd.to_datetime(mark_date), color=colors[2], linestyle='--', linewidth=3)
            ax[i, j].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax[i, j].xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
            ax[i, j].xaxis.grid(True)
            ax[i, j].grid(color='grey', linestyle='-', linewidth=1, alpha=0.1)
            ax[i, j].tick_params(axis='x', labelrotation=90, labelsize=6)

            ax[i, j].set_xlim(df1.index.values[28], df1.index.values[-1])
            ax[i, j].xaxis.set_major_locator(mdates.DayLocator(interval=7))

            ax[i, j].xaxis.grid(True)
            ax[i, j].yaxis.grid(False)
            ax[i, j].legend()

    return plt


if __name__ == '__main__':
    start_date = '2021-02-04'
    df3nv = pd.read_csv(os.path.join('..','predictions', f'vacc_3rdwave_without_vaccination_{start_date}.csv'))
    df3v = pd.read_csv(os.path.join('..','predictions', f'vacc_3rdwave_with_vaccination_{start_date}.csv'))
    df4nv = pd.read_csv(os.path.join('..','predictions', f'vacc_4thwave_without_vaccination_{start_date}.csv'))
    df4v = pd.read_csv(os.path.join('..','predictions', f'vacc_4thwave_with_vaccination_{start_date}.csv'))
    m1 = sum(df3nv.muertes_diarias) - sum(df3v.muertes_diarias)
    m2 = sum(df4nv.muertes_diarias) - sum(df4v.muertes_diarias)
    print('sin cuarta ola: ', m1)
    print('con cuarta ola: ', m2)

    plt = plot_vacc_comparison(df3nv, df3v, mark_date=start_date)
    plt.tight_layout()
    plt.show()

    plt = plot_vacc_comparison(df4nv, df4v, mark_date=start_date)
    plt.tight_layout()
    plt.show()
