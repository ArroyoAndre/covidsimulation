# metric_name: (numerator_measurement_name, Optional[denominator_measurement_name])

METRICS = {
    'population': ('population', None),
    'infected': ('infected', None),
    'in_isolation': ('in_isolation', None),
    'diagnosed': ('diagnosed', None),
    'dead': ('dead', None),
    'dead_confirmed': ('dead_confirmed', None),
    'hospitalized': ('hospitalized', None),
    'in_ventilation': ('in_ventilation', None),
    'in_icu': ('in_icu', None),
    'contagious': ('contagious', None),
    'finished_contagion': ('finished_contagion', None),
    'transmitted': ('transmitted', None),
    'susceptible': ('susceptible', None),
    'in_hospital_bed': ('in_hospital_bed', None),
    'pc_infected': ('infected', 'population'),
    'pc_isolated': ('in_isolation', 'population'),
    'pc_contagious': ('contagious', 'population'),
    'rt': ('transmitted', 'finished_contagion'),
}
