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
    'contagious': ('contagious', None),
    'contagion_ended': ('contagion_ended', None),
    'transmited': ('transmited', None),
    'succeptible': ('succeptible', None),
    'in_hospital_bed': ('in_hospital_bed', None),
    'pc_succeptible': ('succeptible', 'population'),
    'pc_infectados': ('infectados', 'population'),
    'pc_in_isolation': ('in_isolation', 'population'),
    'pc_contagious': ('contagious', 'population'),
    'rt': ('transmited', 'contagion_ended'),
}
