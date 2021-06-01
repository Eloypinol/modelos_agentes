from calc.variables import reset_variables, set_variable
from datetime import datetime, date, timedelta


class VaccinationScenario:

    def __init__(self, id, start_day, interventions=None, variables=None):
        self.id = id
        self.interventions = interventions
        self.variables = variables
        self.start_day = start_day

    def apply(self):
        reset_variables()
        if self.interventions:
            ivs = self.interventions
            set_variable('interventions', ivs)

        if self.variables:
            for key, val in self.variables.items():
                set_variable(key, val)

        set_variable('preset_scenario', self.id)

    def apply_vaccination_stages(self, vaccination_stages):
        v_stages = []
        for stage in vaccination_stages:
            s = stage.copy()
            s['start_date'] = (date.fromisoformat(s['start_date']) - date.fromisoformat(self.start_day)).days
            s['end_date'] = (date.fromisoformat(s['end_date']) - date.fromisoformat(self.start_day)).days
            v_stages.append(s)
        self.variables['vaccination_stages'] = v_stages


class VaccinationScenarioCV(VaccinationScenario):

    def __init__(self, start_date='2020-06-01', vaccination_stages=[]):
        variables = {
            'start_date': start_date,
            'area_name': 'Comunitat Valenciana',
            'vaccines': [
                {
                    'name': 'Pfizer-BioNTech',
                    'id': 1,
                    'two_doses': True,
                    'days_between_doses': 21,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.685, # https://www.nejm.org/doi/full/10.1056/NEJMc2036242
                    'efectiveness_after_second_dose': 0.95
                },
                {
                    'name': 'AstraZeneca',
                    'id': 2,
                    'two_doses': True,
                    'days_between_doses': 84,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.76, # https://www.bmj.com/content/372/bmj.n326
                    'efectiveness_after_second_dose': 0.82
                }
                ],
            'vaccination_stages': vaccination_stages
        }

        interventions = [
            ['limit-mass-gatherings', '2020-06-01', 10],
            ['limit-mass-gatherings', '2020-08-15', 9],
            ['limit-mass-gatherings', '2020-10-25', 6],
            ['set-test-avg-delay', '2020-06-01', 3],
            ['set-test-avg-delay', '2020-09-01', 5],

            ['test-all-with-symptoms', '2020-06-01', 90],
            ['test-all-with-symptoms', '2020-07-15', 85],
            ['test-all-with-symptoms', '2020-08-15', 80],
            ['limit-mobility', '2020-06-01', 58.0],
            ['limit-mobility', '2020-09-01', 63.0],
            ['limit-mobility', '2020-09-15', 63.0],
            ['limit-mobility', '2020-10-10', 55.0],
            ['limit-mobility', '2020-10-15', 45.0],
            ['limit-mobility', '2020-10-25', 49.0],
            ['limit-mobility', '2020-11-16', 66.0],
            ['limit-mobility', '2020-11-30', 5.0],
            ['limit-mobility', '2020-12-05', 2.0],
            ['limit-mobility', '2020-12-15', 0.0],

            ['test-with-contact-tracing', '2020-06-01', 35],
            ['test-with-contact-tracing', '2020-08-01', 40],
            ['test-with-contact-tracing', '2020-08-17', 65],
            ['test-with-contact-tracing', '2020-09-15', 45],
            ['test-with-contact-tracing', '2020-11-15', 50],
            ['test-with-contact-tracing', '2020-12-10', 30],
            ['test-with-contact-tracing', '2020-12-15', 17],

            ['limit-mobility', '2021-01-15', 60.0],
            ['test-with-contact-tracing', '2021-01-15', 60],

            # gradual descent
            ['limit-mobility', '2021-01-30', 38.0],
            ['test-with-contact-tracing', '2021-01-30', 50],


            ['import-infections', '2020-06-01', 75],
        ]

        super().__init__(id='VaccinationScenarioCV', start_day=start_date,
                         interventions=interventions, variables=variables)
        self.apply_vaccination_stages(vaccination_stages)

class VaccinationScenarioCV_updated(VaccinationScenario):

    def __init__(self, start_date='2020-06-01', vaccination_stages=[]):
        variables = {
            'start_date': start_date,
            'area_name': 'Comunitat Valenciana',
            'vaccines': [
                {
                    'name': 'Pfizer-BioNTech',
                    'id': 1,
                    'two_doses': True,
                    'days_between_doses': 21,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.685, # https://www.nejm.org/doi/full/10.1056/NEJMc2036242
                    'efectiveness_after_second_dose': 0.95
                },
                {
                    'name': 'AstraZeneca',
                    'id': 2,
                    'two_doses': True,
                    'days_between_doses': 84,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.76, # https://www.bmj.com/content/372/bmj.n326
                    'efectiveness_after_second_dose': 0.82
                }
                ],
            'vaccination_stages': vaccination_stages
        }

        interventions = [
            ['limit-mass-gatherings', '2020-06-01', 10],
            ['limit-mass-gatherings', '2020-08-15', 9],
            ['limit-mass-gatherings', '2020-10-25', 6],
            ['limit-mass-gatherings', '2020-12-23', 8],
            ['limit-mass-gatherings', '2021-01-07', 6],
            ['set-test-avg-delay', '2020-06-01', 3],
            ['set-test-avg-delay', '2020-09-01', 5],

            ['test-all-with-symptoms', '2020-06-01', 90],
            ['test-all-with-symptoms', '2020-07-15', 85],
            ['test-all-with-symptoms', '2020-08-15', 80],
            ['limit-mobility', '2020-06-01', 58.0],
            ['limit-mobility', '2020-09-01', 63.0],
            ['limit-mobility', '2020-09-15', 63.0],
            ['limit-mobility', '2020-10-10', 55.0],
            ['limit-mobility', '2020-10-15', 45.0],
            ['limit-mobility', '2020-10-25', 49.0],
            ['limit-mobility', '2020-11-16', 66.0],
            ['limit-mobility', '2020-11-30', 5.0],
            ['limit-mobility', '2020-12-05', 0.0],
            ['limit-mobility', '2020-12-15', 0.0],

            ['test-with-contact-tracing', '2020-06-01', 35],
            ['test-with-contact-tracing', '2020-08-01', 40],
            ['test-with-contact-tracing', '2020-08-17', 65],
            ['test-with-contact-tracing', '2020-09-15', 45],
            ['test-with-contact-tracing', '2020-11-15', 50],
            ['test-with-contact-tracing', '2020-12-10', 30],
            ['test-with-contact-tracing', '2020-12-15', 17],

            # Restricciones 07/01/2021 https://www.levante-emv.com/comunitat-valenciana/2021/01/05/nuevas-medidas-restricciones-cierre-pueblos-horarios-bares-coronavirus-27063496.html
            ['limit-mobility', '2021-01-07', 50.0],
            #['test-with-contact-tracing', '2021-01-07', 30],
            ['limit-mass-gatherings', '2021-01-07', 4],

            # Restricciones 21/01/2021 https://www.levante-emv.com/comunitat-valenciana/2021/01/19/nuevas-medidas-restricciones-comunidad-valenciana-coronavirus-valencia-29391219.html
            ['limit-mobility', '2021-01-21', 30.0],

            # Restricciones 01/02/2021 
            ['limit-mass-gatherings', '2021-02-01', 2],

            # Restricciones 04/03/2021 https://www.levante-emv.com/comunitat-valenciana/2021/03/01/desescalada-valencia-restricciones-coronavirus-comunidad-valenciana-35775424.html
            ['limit-mass-gatherings', '2021-03-04', 4],
            ['limit-mobility', '2021-03-04', 20.0],

            # Restricciones 15/03/2021 https://www.levante-emv.com/comunitat-valenciana/2021/03/15/restricciones-valencia-15-marzo-apertura-interior-bares-40997060.html
            ['limit-mass-gatherings', '2021-03-15', 6],
            #['limit-mobility', '2021-03-15', 70.0],

            # Restricciones 12/04/2021 https://www.levante-emv.com/comunitat-valenciana/2021/04/08/asi-quedan-restricciones-valencia-coronavirus-comunidad-valenciana-46177386.html
            ['limit-mobility', '2021-03-04', 15.0],


            # Restricciones 12/04/2021 https://www.levante-emv.com/comunitat-valenciana/2021/04/08/asi-quedan-restricciones-valencia-coronavirus-comunidad-valenciana-46177386.html
            #['limit-mass-gatherings', '2021-02-01', 6],
            #['limit-mobility', '2021-01-07', 45.0],

            # gradual descent
            #['limit-mobility', '2021-01-15', 70.0],
            #['limit-mass-gatherings', '2021-01-07', 4],
            #['limit-mobility', '2021-01-30', 38.0],
            ['test-with-contact-tracing', '2021-01-30', 50],


            ['import-infections', '2020-06-01', 75],
        ]

        super().__init__(id='VaccinationScenarioCV_updated', start_day=start_date,
                         interventions=interventions, variables=variables)
        self.apply_vaccination_stages(vaccination_stages)

class VaccinationScenarioCV_updated_more(VaccinationScenario):

    def __init__(self, start_date='2020-06-01', vaccination_stages=[]):
        variables = {
            'start_date': start_date,
            'area_name': 'Comunitat Valenciana',
            'vaccines': [
                {
                    'name': 'Pfizer-BioNTech',
                    'id': 1,
                    'two_doses': True,
                    'days_between_doses': 21,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.685, # https://www.nejm.org/doi/full/10.1056/NEJMc2036242
                    'efectiveness_after_second_dose': 0.95
                },
                {
                    'name': 'AstraZeneca',
                    'id': 2,
                    'two_doses': True,
                    'days_between_doses': 84,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.76, # https://www.bmj.com/content/372/bmj.n326
                    'efectiveness_after_second_dose': 0.82
                }
                ],
            'vaccination_stages': vaccination_stages
        }

        interventions = [
            ['set-test-avg-delay', '2020-12-01', 5],
            ['set-test-avg-delay', '2021-01-25', 2],

            ['test-all-with-symptoms', '2020-12-01', 60],
            ['test-all-with-symptoms', '2021-02-01', 90],

            ['limit-mobility', '2020-11-01', 25.0],
            ['limit-mobility', '2020-11-15', 35.0],
            ['limit-mobility', '2020-11-30', 35.0],
            ['limit-mobility', '2020-12-01', 10.0],
            ['limit-mobility', '2020-12-15', 30.0],
            ['limit-mobility', '2020-12-30', 30.0],

            ['test-with-contact-tracing', '2020-12-01', 35],
            ['test-with-contact-tracing', '2020-12-10', 15],
            ['test-with-contact-tracing', '2020-12-15', 15],
            ['test-with-contact-tracing', '2020-12-30', 5],
            ['test-with-contact-tracing', '2021-01-31', 90],
            ['test-with-contact-tracing', '2021-03-31', 60],

            # Restricciones 07/01/2021 https://www.levante-emv.com/comunitat-valenciana/2021/01/05/nuevas-medidas-restricciones-cierre-pueblos-horarios-bares-coronavirus-27063496.html
            ['limit-mobility', '2021-01-07', 0.0],
            ['limit-mass-gatherings', '2021-01-07', 4],

            # Restricciones 21/01/2021 https://www.levante-emv.com/comunitat-valenciana/2021/01/19/nuevas-medidas-restricciones-comunidad-valenciana-coronavirus-valencia-29391219.html
            ['limit-mobility', '2021-01-21', 30.0],
            ['limit-mobility', '2021-01-31', 70.0],
            ['limit-mobility', '2021-02-07', 85.0],
            ['limit-mobility', '2021-02-14', 30.0],

            # Restricciones 01/02/2021
            ['limit-mass-gatherings', '2021-02-01', 2],

            # Restricciones 04/03/2021 https://www.levante-emv.com/comunitat-valenciana/2021/03/01/desescalada-valencia-restricciones-coronavirus-comunidad-valenciana-35775424.html
            ['limit-mass-gatherings', '2021-03-04', 4],
            ['limit-mobility', '2021-03-04', 30.0],

            # Restricciones 15/03/2021 https://www.levante-emv.com/comunitat-valenciana/2021/03/15/restricciones-valencia-15-marzo-apertura-interior-bares-40997060.html
            ['limit-mass-gatherings', '2021-03-15', 6],
            #['limit-mobility', '2021-03-15', 70.0],

            # Restricciones 12/04/2021 https://www.levante-emv.com/comunitat-valenciana/2021/04/08/asi-quedan-restricciones-valencia-coronavirus-comunidad-valenciana-46177386.html
            ['limit-mobility', '2021-04-12', 20.0],

            # Fin estado de alarma
            ['limit-mobility', '2021-05-09', 10.0],
            ['limit-mass-gatherings', '2021-05-09', 8],

            ['import-infections', '2020-12-30', 4450],
            ['import-recovered', '2020-12-01', 350000],
        ]

        super().__init__(id='VaccinationScenarioCV_updated', start_day=start_date,
                         interventions=interventions, variables=variables)
        self.apply_vaccination_stages(vaccination_stages)


class VaccinationScenarioWithFourthWaveCV(VaccinationScenario):

    def __init__(self, start_date='2020-06-01', vaccination_stages=[]):
        variables = {
            'start_date': start_date,
            'area_name': 'Comunitat Valenciana',
            'vaccines': [
                {
                    'name': 'Pfizer-BioNTech',
                    'id': 1,
                    'two_doses': True,
                    'days_between_doses': 21,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.685, # https://www.nejm.org/doi/full/10.1056/NEJMc2036242
                    'efectiveness_after_second_dose': 0.95
                },
                {
                    'name': 'AstraZeneca',
                    'id': 2,
                    'two_doses': True,
                    'days_between_doses': 84,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.76, # https://www.bmj.com/content/372/bmj.n326
                    'efectiveness_after_second_dose': 0.82
                }
                ],
            'vaccination_stages': vaccination_stages
        }

        interventions = [
            ['limit-mass-gatherings', '2020-06-01', 10],
            ['limit-mass-gatherings', '2020-08-15', 9],
            ['limit-mass-gatherings', '2020-10-25', 6],
            ['set-test-avg-delay', '2020-06-01', 3],
            ['set-test-avg-delay', '2020-09-01', 5],

            ['test-all-with-symptoms', '2020-06-01', 90],
            ['test-all-with-symptoms', '2020-07-15', 85],
            ['test-all-with-symptoms', '2020-08-15', 80],
            ['limit-mobility', '2020-06-01', 58.0],
            ['limit-mobility', '2020-09-01', 63.0],
            ['limit-mobility', '2020-09-15', 63.0],
            ['limit-mobility', '2020-10-10', 55.0],
            ['limit-mobility', '2020-10-15', 45.0],
            ['limit-mobility', '2020-10-25', 49.0],
            ['limit-mobility', '2020-11-16', 66.0],
            ['limit-mobility', '2020-11-30', 5.0],
            ['limit-mobility', '2020-12-05', 2.0],
            ['limit-mobility', '2020-12-15', 0.0],

            ['test-with-contact-tracing', '2020-06-01', 35],
            ['test-with-contact-tracing', '2020-08-01', 40],
            ['test-with-contact-tracing', '2020-08-17', 65],
            ['test-with-contact-tracing', '2020-09-15', 45],
            ['test-with-contact-tracing', '2020-11-15', 50],
            ['test-with-contact-tracing', '2020-12-10', 30],
            ['test-with-contact-tracing', '2020-12-15', 17],

            ['limit-mobility', '2021-01-15', 60.0],
            ['test-with-contact-tracing', '2021-01-15', 60],

            # steep descent
            ['limit-mobility', '2021-01-30', 45.0],
            ['test-with-contact-tracing', '2021-01-30', 50],

            # Pesimistic future (4th wave)
            ['limit-mobility', '2021-03-31', 30.0],
            ['test-with-contact-tracing', '2021-03-31', 30],
            # ['limit-mobility', '2021-05-15', 50.0],
            # ['test-with-contact-tracing', '2021-05-15', 50],


            ['import-infections', '2020-06-01', 75],
        ]

        super().__init__(id='VaccinationScenarioWithFourthWaveCV', start_day=start_date,
                         interventions=interventions, variables=variables)
        self.apply_vaccination_stages(vaccination_stages)


class VaccinationSimpleScenarioCV(VaccinationScenario):

    def __init__(self, start_date='2020-07-01', vaccination_stages=[]):
        variables = {
            'start_date': start_date,
            'area_name': 'CVDebug1000',
            'vaccines': [
                {
                    'name': 'Pfizer-BioNTech',
                    'id': 1,
                    'two_doses': True,
                    'days_between_doses': 21,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.685, # https://www.nejm.org/doi/full/10.1056/NEJMc2036242
                    'efectiveness_after_second_dose': 0.95
                },
                {
                    'name': 'AstraZeneca',
                    'id': 2,
                    'two_doses': True,
                    'days_between_doses': 84,
                    'days_to_immunization': 7,
                    'efectiveness_after_first_dose': 0.76, # https://www.bmj.com/content/372/bmj.n326
                    'efectiveness_after_second_dose': 0.82
                }
                ],
            'vaccination_stages': vaccination_stages
        }

        interventions = [
            ['import-infections', '2020-07-01', 75],
        ]

        super().__init__(id='VaccinationSimpleScenarioCV', start_day=start_date,
                         interventions=interventions, variables=variables)
        self.apply_vaccination_stages(vaccination_stages)

