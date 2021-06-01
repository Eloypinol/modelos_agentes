import pandas as pd
import numpy as np

# df = pd.read_excel('IDXXX - Solicitud JC 20200520 fallecidos COVID 1.0.xlsx', sheet_name='Sheet1', header=3,
#                    parse_dates=['Fecha de alta'])
# df = df.pivot_table(index='Fecha de alta', columns=['Sexo', 'Grupo edad'], values='Éxitus', aggfunc='sum')
# df_dates = pd.DataFrame({'Fecha de alta': pd.date_range(start='1/2/2020', end='31/12/2020')})
# df = pd.merge(df_dates, df, on='Fecha de alta', how='outer').fillna(0)
# df = df.set_index('Fecha de alta')

df = pd.read_excel('IDXXX - Solicitud JC 20200520 hospitalizados COVID Y NO COVID 1.0.xlsx', sheet_name='Sheet1',
                   header=3, parse_dates=['Fecha de hospitalización'])
df = df.loc[df['Situación COVID'] == 'COVID']

df_attr_real = df.pivot_table(index='Grupo de edad', columns=['Sexo'],
                                        values='Pacientes hospitalizados', aggfunc='sum')

df = df.pivot_table(index='Fecha de hospitalización', columns=['Sexo', 'Grupo de edad'],
                    values='Pacientes hospitalizados', aggfunc='sum')
print(df)



