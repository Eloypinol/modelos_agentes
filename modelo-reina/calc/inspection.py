import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('data/Total de casos COVID a fecha de corte.txt',
                 encoding='utf-16LE', sep='\t', parse_dates=['Fecha de movimiento'], dayfirst=True).fillna(0)

df_dpto1 = sum(df.loc[(df['Número Departamento'] == 3), 'Casos confirmados'])

df = df.drop(columns=['Fecha', 'Fecha desde', 'Fecha hasta',
                      ' MAX(Fecha de movimiento - completa)', 'Código de centro'])

df = df.rename(columns={'Número Departamento': 'Id', 'Centro': 'Nom', 'Casos confirmados': 'Positius_diaris',
                        'Ingresos confirmados': 'Hospitalitzats_diaris', 'Casos confirmados en UCI': 'UCI',
                        'Fallecidos': 'Morts_diaris', 'Altas': 'Recuperats_diaris', 'Fecha de movimiento': 'Data'})
df = df.sort_values(['Data', 'Id'], ascending=[True, True]).reset_index(drop=True)
# df.to_csv('data/series_temporales_por_dpto.csv', index=None)
# exit()
df['Positius'] = df.groupby(['Id', 'Nom'])['Positius_diaris'].transform(pd.Series.cumsum)
df['Recuperats'] = df.groupby(['Id', 'Nom'])['Recuperats_diaris'].transform(pd.Series.cumsum)
df['Hospitalitzats'] = df.groupby(['Id', 'Nom'])['Hospitalitzats_diaris'].transform(pd.Series.cumsum)
df['Morts'] = df.groupby(['Id', 'Nom'])['Morts_diaris'].transform(pd.Series.cumsum)
df['Hospitalitzats'] = df['Hospitalitzats'] - df['Recuperats_diaris'] - df['Morts_diaris']


df_totals = df.loc[df['Data'] == df['Data'].max(), ['Id', 'Nom', 'Morts', 'Positius', 'Positius_diaris']]
print(df_totals)
print(f'Total fallecidos: {sum(df_totals["Morts"])}')
print(f'Total confirmados: {sum(df_totals["Positius"])}')
print(sum(df_totals['Positius_diaris']))

df = df.loc[df['Id'] != 99]
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=200)
ax = sns.lineplot(x="Data", y="Recuperats_diaris", hue='Nom', data=df)
plt.show()

df.to_csv('data/series_temporales_por_dpto.csv', index=None)