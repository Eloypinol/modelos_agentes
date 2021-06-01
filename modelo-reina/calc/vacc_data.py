import pandas as pd
import numpy as np
from datetime import datetime


def get_vaccination_stages_so_far(vaccine='Pfizer-BioNTech'):

    def dosis_to_int(str_dosis):
        if len(str_dosis) > 0:
            return int(str_dosis.replace('.', ''))
        return 0

    URL = 'https://raw.githubusercontent.com/civio/covid-vaccination-spain/main/data.csv'
    df = pd.read_csv(URL, converters={'dosis Pfizer': dosis_to_int, 'dosis Moderna': dosis_to_int,
                                      'dosis AstraZeneca': dosis_to_int, 'dosis Janssen': dosis_to_int,
                                      'dosis entregadas': dosis_to_int, 'dosis administradas': dosis_to_int,
                                      'informe': lambda x: datetime.strptime(x, '%d/%M/%Y').strftime('%Y-%M-%d'),
                                      'personas con pauta completa': dosis_to_int}
                     ).drop(['% sobre entregadas', 'última vacuna registrada'], axis=1)
    df = df.loc[df['comunidad autónoma'] == 'C. Valenciana'].drop(['comunidad autónoma'], axis=1)
    df = pd.concat([pd.DataFrame([{'informe': '2021-01-03'}]), df], ignore_index=True).fillna(0)
    df.index = range(len(df.index))
    df = df.astype({'dosis Pfizer': int, 'dosis Moderna': int,
                    'dosis AstraZeneca': int, 'dosis Janssen': int,
                    'dosis entregadas': int, 'dosis administradas': int,
                    'informe': str, 'personas con pauta completa': int})

    df['Pfizer/Moderna'] = df['dosis Pfizer'] + df['dosis Moderna']
    df['AZ/Janssen'] = df['dosis AstraZeneca'] + df['dosis Janssen']
    df['rel_AZ'] = df['AZ/Janssen'] / ( df['AZ/Janssen'] + df['Pfizer/Moderna'])
    df.loc[0,"rel_AZ"] = 0 #

    dosis1 = np.zeros(shape=(df.shape[0]), dtype=int)
    dosis2 = np.zeros(shape=(df.shape[0]), dtype=int)
    dosis1[1:] = df['dosis administradas'].iloc[1:].values - df['dosis administradas'].iloc[:-1].values
    dosis2[1:] = df['personas con pauta completa'].iloc[1:].values - df['personas con pauta completa'].iloc[:-1].values
    dosis1 -= dosis2
    for i in range(dosis1.shape[0] - 1):
        if dosis1[i] < 0:
            dosis1[i + 1] = dosis1[i] + dosis1[i + 1]
            dosis1[i] = 0
    df['dosis 1'] = dosis1
    df['dosis 2'] = dosis2

    df['dosis 1 AZ'] = (df['dosis 1'] + df['dosis 2']) * df['rel_AZ']
    df['dosis 1 AZ'] = df['dosis 1 AZ'].round(0).astype(int)
    df['dosis 1 Pfizer'] = df['dosis 1'] - df['dosis 1 AZ']
    num = df._get_numeric_data()
    num[num < 0] = 0
    df.to_csv("vacunas.csv")
    vacc_stages = []
    total = 0
    for i in range(df.shape[0]):
        new_doses2 = df['dosis 2'].iloc[i]
        new_doses1Pf = df['dosis 1 Pfizer'].iloc[i]
        new_doses1AZ = df['dosis 1 AZ'].iloc[i]
        datevacc = df['informe'].iloc[i]
        ####### Primera etapa de vacunacion
        if datetime.fromisoformat(datevacc) < datetime.fromisoformat('2021-03-01'):
            # Usuarios de residencias   
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 1, 'min-age': 80, 'max_age': 100, 'rate': int(new_doses1Pf*0.5)
            })
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 2, 'min-age': 80, 'max_age': 100, 'rate': int(new_doses2*0.5)
            })
            # Personal sanitario
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 1, 'min-age': 25, 'max_age': 65, 'rate': int(new_doses1Pf*0.5)
            })
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 2, 'min-age': 25, 'max_age': 65, 'rate': int(new_doses2*0.5)
            })

            # Profesionales riesgo
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'AstraZeneca',
                'dose': 1, 'min-age': 25, 'max_age': 65, 'rate': new_doses1AZ
            })
        ###### Segunda etapa de vacunacion
        elif datetime.fromisoformat(datevacc) < datetime.fromisoformat('2021-04-09'):
            # Mayores de 70     
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 1, 'min-age': 70, 'max_age': 100, 'rate': new_doses1Pf
            })
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 2, 'min-age': 70, 'max_age': 100, 'rate': new_doses2
            })

            # Profesionales riesgo
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'AstraZeneca',
                'dose': 1, 'min-age': 25, 'max_age': 65, 'rate': new_doses1AZ
            })
        ###### Segunda etapa de vacunacion - problemas con astrazeneca
        elif datetime.fromisoformat(datevacc) < datetime.fromisoformat('2021-05-10'):
            # Mayores de 70     
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 1, 'min-age': 70, 'max_age': 100, 'rate': new_doses1Pf
            })
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'Pfizer-BioNTech',
                'dose': 2, 'min-age': 70, 'max_age': 100, 'rate': new_doses2
            })

            # AZ y Janssen a mayores de 60
            vacc_stages.append({
                'start_date': df['informe'].iloc[i], 'end_date': df['informe'].iloc[i], 'vaccine': 'AstraZeneca',
                'dose': 1, 'min-age': 60, 'max_age': 70, 'rate': new_doses1AZ
            })                   

        total = total + new_doses1Pf + new_doses1AZ + new_doses2
    print(total)
    return vacc_stages


if __name__ == '__main__':
    stages = get_vaccination_stages_so_far()
    [print(stage) for stage in stages]
