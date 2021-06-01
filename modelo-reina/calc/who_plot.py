import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

ct_curves = [0, 40, 80, 100]
dfs = {}
for ct in ct_curves:
    dfs[ct] = pd.read_csv(f'../WHO_model_reina_out_del03_ct{ct:03d}.csv', parse_dates=['Fecha'], dayfirst=True)

# 'Fecha', 'Casos activos COVID-19', 'Casos acumulados COVID-19',
#        'Nuevos casos diarios COVID-19', 'Personas recuperadas',
#        'Casos detectados COVID-19', 'Casos detectados acumulados',
#        'Personas hospitalizadas por COVID-19', 'Ingresados en UCI',
#        'Número total de fallecidos'
cmap = plt.get_cmap('tab20b')
c_shift = 0
cut_date = '2020-05-30'

plots = [
    ('Número total de fallecidos', 'Contact tracing impact on number of dead'),
    ('Casos detectados COVID-19', 'Contact tracing impact on number of detected cases'),
    ('Personas hospitalizadas por COVID-19', 'Contact tracing impact on number of hospitalized'),
    ('Casos activos COVID-19', 'Contact tracing impact on prevalence'),
]

plot = 3

fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
for i, ct in enumerate(dfs):
    if i == 0:
        date = dfs[ct]['Fecha']
        y = dfs[ct][plots[plot][0]]
        ax.plot(date, y, c=cmap(c_shift), alpha=0.5, linewidth=3, label=f'0%')
    else:
        df_after = dfs[ct].loc[dfs[ct]['Fecha'] >= np.datetime64(cut_date)]
        date = df_after['Fecha']
        y = df_after[plots[plot][0]]
        if ct == 80:
            ax.plot(date, y, c=cmap(c_shift + 4), alpha=1.0, linestyle='--', label=f'{ct}%')
        else:
            ax.plot(date, y, c=cmap(c_shift + i), alpha=1.0, linestyle='--', label=f'{ct}%')

ax.axvline(datetime.strptime('2020-05-18', '%Y-%m-%d'), color='k', linestyle='dashed', alpha=0.4, linewidth=4)
ax.text(datetime.strptime('2020-05-17', '%Y-%m-%d'), ax.get_ylim()[1] * 0.95, 'Start of CT',
        {'ha': 'right', 'va': 'top'}, rotation=90, size=12, alpha=0.75)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
# ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
ax.tick_params(axis='x', labelrotation=90, labelsize=7)
ax.legend()
plt.title(plots[plot][1])
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
for i, ct in enumerate(dfs):
    if i == 0:
        date = dfs[ct]['Fecha']
        ax.plot(date, dfs[ct][plots[1][0]], c=cmap(c_shift), alpha=0.5, linewidth=5, label='detected without CT')
        ax.plot(date, dfs[ct][plots[0][0]], c=cmap(c_shift + 4), alpha=0.5, linewidth=5, label='dead without CT')
    else:
        df_after = dfs[ct].loc[dfs[ct]['Fecha'] >= np.datetime64(cut_date)]
        date = df_after['Fecha']
        if ct == 40:
            ax.plot(date, df_after[plots[1][0]], c=cmap(17), alpha=1.0, linewidth=3, linestyle='dashed', label=f'detected with {ct}% CT')
            ax.plot(date, df_after[plots[0][0]], c=cmap(17), alpha=1.0, linewidth=3, linestyle='dashdot', label=f'dead with {ct}% CT')
        else:
            ax.plot(date, df_after[plots[1][0]], c=cmap(c_shift + i), alpha=1.0, linewidth=3, linestyle='dashed', label=f'detected with {ct}% CT')
            ax.plot(date, df_after[plots[0][0]], c=cmap(c_shift + i + 4), alpha=1.0, linewidth=3, linestyle='dashdot', label=f'dead with {ct}% CT')

ax.axvline(datetime.strptime('2020-05-18', '%Y-%m-%d'), color='k', linestyle='dashed', alpha=0.4, linewidth=4)
ax.text(datetime.strptime('2020-05-17', '%Y-%m-%d'), ax.get_ylim()[1] * 0.95, 'Start of CT',
        {'ha': 'right', 'va': 'top'}, rotation=90, size=12, alpha=0.75)


ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
# ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.15)
ax.tick_params(axis='x', labelrotation=90, labelsize=7)
ax.legend()
plt.title('Impact of Contact Tracing (CT) on the number of COVID-19 detected cases and deaths')
plt.show()
