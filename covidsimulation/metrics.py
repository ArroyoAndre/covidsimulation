# metric_name: (numerator_measurement_name, Optional[denominator_measurement_name])

METRICS = {
    'population': ('population', None),
    'infected': ('infected', None),
    'in_isolation': ('in_isolation', None),
    'diagnosed': ('diagnosed', None),
    'deaths': ('deaths', None),
    'confirmed_deaths': ('confirmed_deaths', None),
    'inpatients': ('inpatients', None),
    'ventilated': ('ventilated', None),
    'in_intensive_care': ('in_intensive_care', None),
    'in_hospital_bed': ('in_hospital_bed', None),
    'contagious': ('contagious', None),
    'contagion_ended': ('contagion_ended', None),
    'transmitted': ('transmitted', None),
    'susceptible': ('susceptible', None),
    'pc_susceptible': ('susceptible', 'population'),
    'pc_infected': ('infected', 'population'),
    'pc_in_isolation': ('in_isolation', 'population'),
    'pc_contagious': ('contagious', 'population'),
    'rt': ('transmited', 'contagion_ended'),
}
