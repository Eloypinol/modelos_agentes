import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, date, timedelta
from calc.variables import get_variable

df_real_data = pd.read_csv('data/cv_retrieved_daily.csv', parse_dates=['Data'], index_col=['Data'], dayfirst=True)
df_real_data_ucis = df_real_data['UCI']
df_real_data_activos = df_real_data['Casos_Actius']
df_real_data_hospitalizados = df_real_data['Hospitalitzats']
df_real_data_infectados = df_real_data['Casos_Actius']
df_real_data_fallecidos = df_real_data['Morts']

df_del16ct000 = pd.read_csv('model_reina_out_del16_ct00.csv', parse_dates=['date'], index_col=['date'])
df_del16ct010 = pd.read_csv('model_reina_out_del16_ct10.csv', parse_dates=['date'], index_col=['date'])
df_del16ct020 = pd.read_csv('model_reina_out_del16_ct20.csv', parse_dates=['date'], index_col=['date'])
df_del16ct030 = pd.read_csv('model_reina_out_del16_ct30.csv', parse_dates=['date'], index_col=['date'])
df_del16ct040 = pd.read_csv('model_reina_out_del16_ct40.csv', parse_dates=['date'], index_col=['date'])
df_del16ct050 = pd.read_csv('model_reina_out_del16_ct50.csv', parse_dates=['date'], index_col=['date'])
df_del16ct100 = pd.read_csv('model_reina_out_del16_ct100.csv', parse_dates=['date'], index_col=['date'])
df_del05ct000 = pd.read_csv('model_reina_out_del05_ct00.csv', parse_dates=['date'], index_col=['date'])
df_del05ct010 = pd.read_csv('model_reina_out_del05_ct10.csv', parse_dates=['date'], index_col=['date'])
df_del05ct020 = pd.read_csv('model_reina_out_del05_ct20.csv', parse_dates=['date'], index_col=['date'])
df_del05ct030 = pd.read_csv('model_reina_out_del05_ct30.csv', parse_dates=['date'], index_col=['date'])
df_del05ct040 = pd.read_csv('model_reina_out_del05_ct40.csv', parse_dates=['date'], index_col=['date'])
df_del05ct050 = pd.read_csv('model_reina_out_del05_ct50.csv', parse_dates=['date'], index_col=['date'])
df_del05ct100 = pd.read_csv('model_reina_out_del05_ct100.csv', parse_dates=['date'], index_col=['date'])

dfs = [df_del16ct000, df_del16ct010, df_del16ct020, df_del16ct030, df_del16ct040, df_del16ct050, df_del16ct100,
       df_del05ct000, df_del05ct010, df_del05ct020, df_del05ct030, df_del05ct040, df_del05ct050, df_del05ct100
       ]

dfs_del16 = [df_del16ct000, df_del16ct010, df_del16ct020, df_del16ct030, df_del16ct040, df_del16ct050, df_del16ct100]
dfs_del05 = [df_del05ct000, df_del05ct010, df_del05ct020, df_del05ct030, df_del05ct040, df_del05ct050, df_del05ct100]

df_ratio = df_del16ct000[['ratio_detected']].copy()
df_ratio['ratio_detected'] = 1

ratio_intervals = [
    {'start': '2020-01-01', 'duration': 0, 'from': 0.095, 'to': None},
    {'start': '2020-04-06', 'duration': 28, 'from': 0.095, 'to': 1 / 8},
]

for interval in ratio_intervals:
    start_date = interval['start']
    if interval['duration'] == 0:
        end_date = str(df_ratio.index[-1]).split(' ')[0]
    else:
        end_date = str(np.datetime64(start_date) + np.timedelta64(interval['duration'], 'D'))

    if interval['to'] is None:
        rs = interval['from']
    else:
        rs = np.arange(interval['from'], interval['to'], (interval['to'] - interval['from']) / interval['duration'])

    df_ratio.loc[(df_ratio.index > start_date) & (df_ratio.index <= end_date), 'ratio_detected'] = rs
    df_ratio.loc[(df_ratio.index > end_date), 'ratio_detected'] = interval['to']

for df in dfs:
    df['detected'] = df['infected'] * df['ratio_detected']
    df['all_detected'] = df['all_infected'] * df['ratio_detected']
    df['hospitalizados'] = 0.023 * df['infected']
    df['uci'] = 0.20 * df['hospitalizados']
    df['fallecidos'] = (df['all_infected'] / 80)

    print(np.max(df.loc[(df.index > '2020-05-25'), 'uci']))

# exit()

# date,cases,deaths,icu,recovered,hospitalized
# df_real = pd.read_csv('data/cv_covid19.csv', parse_dates=['date'], index_col=['date'])

fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

detection_delay = 6
dead_delay = 12
real_cases_nuria_dates_shifted = df_real_data_activos.index.values - np.timedelta64(detection_delay, 'D')
model_dead_dates_shifted = df_ratio.index.values + np.timedelta64(dead_delay, 'D')

dfs = dfs_del16
delay = 16
colors = sns.color_palette("Blues", 7)

ax.plot(dfs[0].index.values, dfs[0].detected, alpha=1.00, color=colors[6], linestyle='-', label='infectious whithout contact-tracing')
ax.plot(dfs[0].index.values, dfs[1].detected, alpha=0.80, color=colors[5], linestyle='-', label='infectious 10% contact-tracing')
ax.plot(dfs[0].index.values, dfs[2].detected, alpha=0.80, color=colors[4], linestyle='-', label='infectious 20% contact-tracing')
ax.plot(dfs[0].index.values, dfs[3].detected, alpha=0.80, color=colors[3], linestyle='-', label='infectious 30% contact-tracing')
ax.plot(dfs[0].index.values, dfs[4].detected, alpha=0.80, color=colors[2], linestyle='-', label='infectious 40% contact-tracing')
ax.plot(dfs[0].index.values, dfs[5].detected, alpha=0.80, color=colors[1], linestyle='-', label='infectious 50% contact-tracing')
ax.plot(dfs[0].index.values, dfs[6].detected, alpha=0.80, color=colors[0], linestyle='-', label='infectious 100% contact-tracing')

ax.plot(df_real_data_activos.index.values, df_real_data_activos, alpha=0.50, color='b', linestyle='--',
        label='real infectious')
ax.plot(real_cases_nuria_dates_shifted, df_real_data_activos, alpha=0.25, color='b', linestyle='--',
        label='real infectious shifted')

ax.set_title(f'Contact tracing effect on infectous cases. Delay in detection = {delay} days')


ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))

ax.xaxis.grid(True)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
ax.tick_params(axis='x', labelrotation=90, labelsize=6)
ax.set_xlim(dfs[0].index.values[20], dfs[0].index.values[-1])

y_lim = np.ceil(np.max(dfs[0].detected) / 2500) * 2500
ax.set_ylim(0, 15000)
# ax.set_ylim(0, y_lim)

height = ax.get_ylim()[1]
for intervention in get_variable('interventions')[2:]:
    if intervention[0] in ['limit-mobility', 'test-with-contact-tracing']:
        ax.axvline(datetime.strptime(intervention[1], '%Y-%m-%d'), color='k', linestyle='-', alpha=0.4)
        ax.text(datetime.strptime(intervention[1], '%Y-%m-%d'), height, f'{intervention[0]}',
                {'ha': 'right', 'va': 'top'}, rotation=90, size=8, alpha=0.75)

ax.xaxis.grid(True)
ax.yaxis.grid(False)
ax.legend()
plt.show()

# ############################################################# DEATHS
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

detection_delay = 6
dead_delay = 12
real_cases_nuria_dates_shifted = df_real_data_activos.index.values - np.timedelta64(detection_delay, 'D')
model_dead_dates_shifted = df_ratio.index.values + np.timedelta64(dead_delay, 'D')

colors = sns.color_palette("Reds", 7)

ax.plot(model_dead_dates_shifted, dfs[0].fallecidos, alpha=1.00, color=colors[6], linestyle='-', label='dead whithout contact-tracing')
ax.plot(model_dead_dates_shifted, dfs[1].fallecidos, alpha=1.00, color=colors[5], linestyle='-', label='dead 10% contact-tracing')
ax.plot(model_dead_dates_shifted, dfs[2].fallecidos, alpha=1.00, color=colors[4], linestyle='-', label='dead 20% contact-tracing')
ax.plot(model_dead_dates_shifted, dfs[3].fallecidos, alpha=1.00, color=colors[3], linestyle='-', label='dead 30% contact-tracing')
ax.plot(model_dead_dates_shifted, dfs[4].fallecidos, alpha=1.00, color=colors[2], linestyle='-', label='dead 40% contact-tracing')
ax.plot(model_dead_dates_shifted, dfs[5].fallecidos, alpha=1.00, color=colors[1], linestyle='-', label='dead 50% contact-tracing')
ax.plot(model_dead_dates_shifted, dfs[6].fallecidos, alpha=1.00, color=colors[0], linestyle='-', label='dead 100% contact-tracing')
ax.plot(df_real_data.index.values, df_real_data_fallecidos, color=colors[5], linestyle='--', alpha=0.50, label='real dead')

ax.set_title(f'Contact tracing effect on deaths. Delay in detection = {delay} days')


ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))

ax.xaxis.grid(True)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
ax.tick_params(axis='x', labelrotation=90, labelsize=6)
ax.set_xlim(dfs[0].index.values[20], dfs[0].index.values[-1])

y_lim = np.ceil(np.max(dfs[0].fallecidos) / 2500) * 2500
ax.set_ylim(0, 10000)
# ax.set_ylim(0, y_lim)

height = ax.get_ylim()[1]
for intervention in get_variable('interventions')[2:]:
    if intervention[0] in ['limit-mobility', 'test-with-contact-tracing']:
        ax.axvline(datetime.strptime(intervention[1], '%Y-%m-%d'), color='k', linestyle='-', alpha=0.4)
        ax.text(datetime.strptime(intervention[1], '%Y-%m-%d'), height, f'{intervention[0]}',
                {'ha': 'right', 'va': 'top'}, rotation=90, size=8, alpha=0.75)

ax.xaxis.grid(True)
ax.yaxis.grid(False)
ax.legend()
plt.show()

# ################################################################## ICUS
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

detection_delay = 6
dead_delay = 12
real_cases_nuria_dates_shifted = df_real_data_activos.index.values - np.timedelta64(detection_delay, 'D')
model_dead_dates_shifted = df_ratio.index.values + np.timedelta64(dead_delay, 'D')

colors = sns.color_palette("cubehelix", 8)

ax.plot(dfs[0].index.values, dfs[0].uci, alpha=1.00, color=colors[0], linestyle='-', label='ICUs whithout contact-tracing')
ax.plot(dfs[0].index.values, dfs[1].uci, alpha=1.00, color=colors[1], linestyle='-', label='ICUs 10% contact-tracing')
ax.plot(dfs[0].index.values, dfs[2].uci, alpha=1.00, color=colors[2], linestyle='-', label='ICUs 20% contact-tracing')
ax.plot(dfs[0].index.values, dfs[3].uci, alpha=1.00, color=colors[3], linestyle='-', label='ICUs 30% contact-tracing')
ax.plot(dfs[0].index.values, dfs[4].uci, alpha=1.00, color=colors[4], linestyle='-', label='ICUs 40% contact-tracing')
ax.plot(dfs[0].index.values, dfs[5].uci, alpha=1.00, color=colors[5], linestyle='-', label='ICUs 50% contact-tracing')
ax.plot(dfs[0].index.values, dfs[6].uci, alpha=1.00, color=colors[6], linestyle='-', label='ICUs 100% contact-tracing')

ax.plot(df_real_data_ucis.index.values, df_real_data_ucis, alpha=1.00, color='m', linestyle='--', label='real icu')

ax.set_title(f'Contact tracing effect on ICUs. Delay in detection = {delay} days')

ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))

ax.xaxis.grid(True)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
ax.tick_params(axis='x', labelrotation=90, labelsize=6)
ax.set_xlim(dfs[0].index.values[20], dfs[0].index.values[-1])

y_lim = np.ceil(np.max(dfs[0].uci) / 250) * 250
ax.set_ylim(0, y_lim)

height = ax.get_ylim()[1]
for intervention in get_variable('interventions')[2:]:
    if intervention[0] in ['limit-mobility', 'test-with-contact-tracing']:
        ax.axvline(datetime.strptime(intervention[1], '%Y-%m-%d'), color='k', linestyle='-', alpha=0.4)
        ax.text(datetime.strptime(intervention[1], '%Y-%m-%d'), height, f'{intervention[0]}',
                {'ha': 'right', 'va': 'top'}, rotation=90, size=8, alpha=0.75)

ax.xaxis.grid(True)
ax.yaxis.grid(False)
ax.legend()
plt.show()
