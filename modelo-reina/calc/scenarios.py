from calc.variables import reset_variables, set_variable, get_variable


class Scenario:

    def __init__(self, id, interventions=None, variables=None):
        self.id = id
        self.interventions = interventions
        self.variables = variables

    def apply(self):
        reset_variables()
        if self.interventions:
            ivs = self.interventions
            set_variable('interventions', ivs)

        if self.variables:
            for key, val in self.variables.items():
                set_variable(key, val)

        set_variable('preset_scenario', self.id)


class DefaultScenario(Scenario):

    def __init__(self):
        super().__init__(
            id='default',
            interventions=[],
            variables=[]
        )


class DefaultScenarioCV(Scenario):

    def __init__(self):
        super().__init__(
            id='default',
            interventions=[
                ['test-only-severe-symptoms', '2020-02-28', 70],
                # ['limit-mass-gatherings', '2020-03-12', 5],
                ['set-test-avg-delay', '2020-02-28', 16],
                ['set-test-avg-delay', '2020-05-25', 3],

                ['test-all-with-symptoms', '2020-03-20', 75],
                ['test-all-with-symptoms', '2020-04-04', 80],
                ['test-all-with-symptoms', '2020-04-18', 85],
                ['test-all-with-symptoms', '2020-05-01', 90],
                # ['test-all-with-symptoms', '2020-05-15', 95],
                # ['test-all-with-symptoms', '2020-06-01', 100],

                ['limit-mobility', '2020-03-14', 77],  # lockdown
                ['limit-mobility', '2020-03-28', 81],  # total lockdown
                ['limit-mobility', '2020-04-13', 77.5],  # lockdown
                ['limit-mobility', '2020-04-26', 77.0],  # lockdown + children
                ['limit-mobility', '2020-05-02', 76.5],  # lockdown + children + sport
                ['limit-mobility', '2020-05-04', 76.0],  # stage 0
                ['limit-mobility', '2020-05-18', 75.50],  # stage 1
                ['limit-mobility', '2020-06-01', 76.00],  # stage 2
                # ['limit-mobility', '2020-06-15', 74.00],  # stage 3
                # ['limit-mobility', '2020-06-29', 73.5],  # stage 4

                ['test-with-contact-tracing', '2020-05-18', 70],
                ['import-infections', '2020-02-17', 2],
            ],
            variables={
                'p_infection': 30.0,
                'simulation_days': 326, # hasta 31/12
                # 'simulation_days': 142,  # hasta 30/6
            }
        )


class SecondWaveScenarioCV(Scenario):

    def __init__(self):
        super().__init__(
            id='default',
            interventions=[
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

                # Optimistic future
                ['limit-mobility', '2021-01-25', 69.0],
                ['test-with-contact-tracing', '2021-01-25', 90],
                ['limit-mobility', '2021-02-05', 80.0],
                ['test-with-contact-tracing', '2021-02-05', 100],
                # ['limit-mobility', '2021-02-15', 40.0],
                # ['test-with-contact-tracing', '2021-02-15', 40],
                # ['limit-mobility', '2021-03-15', 30.0],
                # ['test-with-contact-tracing', '2021-03-15', 30],


                ['import-infections', '2020-06-01', 75],
            ],
            variables={
                'start_date': '2020-06-01',
                'area_name': 'Comunitat Valenciana',
                'simulation_days': 188,
            }
        )


class MitigationScenario(Scenario):

    def __init__(self):
        super().__init__(
            id='mitigation',
            interventions=[
                ['test-only-severe-symptoms', '2020-02-28', 70],
                # ['limit-mass-gatherings', '2020-03-12', 5],
                ['set-test-avg-delay', '2020-02-28', 16],
                ['set-test-avg-delay', '2020-05-25', 5],

                ['test-all-with-symptoms', '2020-03-20', 75],
                ['test-all-with-symptoms', '2020-04-04', 80],
                ['test-all-with-symptoms', '2020-04-18', 85],
                ['test-all-with-symptoms', '2020-05-01', 90],
                ['test-all-with-symptoms', '2020-05-15', 95],

                ['limit-mobility', '2020-03-14', 77],  # lockdown
                ['limit-mobility', '2020-03-28', 81],  # total lockdown
                ['limit-mobility', '2020-04-13', 77],  # lockdown
                ['limit-mobility', '2020-04-26', 76.5],  # lockdown + children
                ['limit-mobility', '2020-05-02', 76],  # lockdown + children + sport
                ['limit-mobility', '2020-05-04', 75.5],  # stage 0
                ['limit-mobility', '2020-05-18', 75.0],  # stage 1
                # ['limit-mobility', '2020-06-01', 74.5],  # stage 2
                # ['limit-mobility', '2020-06-15', 74],  # stage 3
                # ['limit-mobility', '2020-06-30', 73.5],  # stage 4

                ['test-with-contact-tracing', '2020-05-18', 40],
                ['import-infections', '2020-02-17', 2],
            ],
            variables={}
        )


class SummerEasingScenario(Scenario):

    def __init__(self):
        super().__init__(
            id='summer-boogie',
            interventions=['limit-mobility', '2020-05-15', 30],
            variables=[]
        )


class HammerDanceScenario(Scenario):

    def __init__(self):
        super().__init__(
            id='hammer-and-dance',
            interventions=[
                ['test-with-contact-tracing', '2020-05-01', 30],
                ['test-with-contact-tracing', '2020-06-01', 40],
                ['test-with-contact-tracing', '2020-07-01', 50],
                ['test-with-contact-tracing', '2020-08-01', 60],
                ['limit-mobility', '2020-05-01', 30],
                ['limit-mobility', '2020-06-24', 25],
                ['limit-mobility', '2020-08-15', 10],
                ['limit-mobility', '2020-12-06', 15],
            ],
            variables=[]
        )


class RetrospectiveEasingScenario(Scenario):

    def __init__(self):
        super().__init__(
            id='looser-restrictions-to-start-with',
            interventions=[],
            variables=[]
        )

    def apply(self):
        super().apply()

        ivs = get_variable('interventions')
        out = []
        for iv in ivs:
            iv = list(iv)
            if iv[0] == 'limit-mobility':
                iv[2] = iv[2] // 2
            out.append(iv)
        set_variable('interventions', out)


SCENARIOS = [
    DefaultScenario(),
    SummerEasingScenario(),
    MitigationScenario(),
    HammerDanceScenario(),
    RetrospectiveEasingScenario(),
]
