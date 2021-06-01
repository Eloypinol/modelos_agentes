import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, date, timedelta
from calc.variables import get_variable

dead_by_dp = (pd.read_csv('data/Historico_Fallecidos_acc_dept_Salud.csv').drop(columns=['Departamento']).to_numpy(dtype=int)[:, -1])

df1 = pd.read_csv('data/cv_mortality_age_sex.csv')
df2 = pd.read_csv('data/cv_deaths_by_dpt.csv', sep=';')
df_pop_ds = pd.read_csv('data/cv_pop_sex_age_ds.csv')

men_80_by_ds = np.sum((df_pop_ds.loc[(df_pop_ds['Sexo'] == 'H') & (df_pop_ds['Edad'] >= 80)]).iloc[:, 3:].to_numpy(dtype=int), axis=0)

n_men = sum(df1.iloc[:10, 4])
n_women = sum(df1.iloc[10:, 4])
total = sum(df1.iloc[:, 4])
ratio_men = n_men / total
ratio_older80_men = sum(df1.iloc[8:10, 3]) / 100

n_men80 = sum(men_80_by_ds)
deads_older80_men_cv = n_men * ratio_older80_men
deads_older80_men_by_dpt = (df2.iloc[:, 1] * ratio_men * ratio_older80_men).to_numpy()
infected_older80_men_by_dpt = deads_older80_men_by_dpt / 0.148
infected_older80_men_cv = deads_older80_men_cv / 0.148

tot_pop = sum(df2.iloc[:, 3])
ratio_older80_men_pop = n_men80 / tot_pop

older80_men_pop_by_dpto = df2.iloc[:, 3] * ratio_older80_men_pop

ratio_older80_men_infected_by_dpto1 = infected_older80_men_by_dpt / older80_men_pop_by_dpto

ratio_older80_men_infected_by_dpto2 = infected_older80_men_by_dpt / men_80_by_ds

prevalence_by_dpto1 = (ratio_older80_men_infected_by_dpto1 * df2.iloc[:, 3]).to_numpy(dtype=int)
prevalence_by_dpto2 = (ratio_older80_men_infected_by_dpto2 * df2.iloc[:, 3]).to_numpy(dtype=int)
df2['prevalencia1'] = prevalence_by_dpto1
df2['prevalencia1_%'] = 100 * prevalence_by_dpto1 / df2.iloc[:, 3]
df2['prevalencia2'] = prevalence_by_dpto2
df2['prevalencia2_%'] = 100 * prevalence_by_dpto2 / df2.iloc[:, 3]

result = df2[['Departamento de Salud', 'prevalencia1', 'prevalencia1_%', 'prevalencia2', 'prevalencia2_%']].copy()
result.to_csv('prevalencia_por_dpto.csv')
