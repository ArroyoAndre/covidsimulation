# metric_name: (numerator_measurement_name, Optional[denominator_measurement_name])

METRICS = {
    'population': ('population', None),
    'infected': ('infected', None),
    'in_isolation': ('in_isolation', None),
    'diagnosed': ('diagnosed', None),
    'death': ('death', None),
    'confirmed_death': ('confirmed_death', None),
    'hospitalized': ('hospitalized', None),
    'in_ventilator': ('in_ventilator', None),
    'in_icu': ('in_icu', None),
    'in_hospital_bed': ('in_hospital_bed', None),
    'contagious': ('contagious', None),
    'finished_contagion': ('finished_contagion', None),
    'transmitted': ('transmitted', None),
    'susceptible': ('susceptible', None),
    'pc_infected': ('infected', 'population'),
    'pc_isolated': ('in_isolation', 'population'),
    'pc_contagious': ('contagious', 'population'),
    'rt': ('transmitted', 'finished_contagion'),
}
