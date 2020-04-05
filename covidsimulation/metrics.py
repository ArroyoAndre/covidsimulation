# metric_name: (numerator_measurement_name, Optional[denominator_measurement_name])

METRICS = {
    'populacao': ('populacao', None),
    'infectados': ('infectados', None),
    'in_isolation': ('in_isolation', None),
    'diagnosticados': ('diagnosticados', None),
    'mortos': ('mortos', None),
    'mortos_confirmados': ('mortos_confirmados', None),
    'internados': ('internados', None),
    'ventilados': ('ventilados', None),
    'em_uti': ('em_uti', None),
    'em_contagio': ('em_contagio', None),
    'contagio_finalizado': ('contagio_finalizado', None),
    'transmitidos': ('transmitidos', None),
    'suceptivel': ('suceptivel', None),
    'in_hospital_bed': ('in_hospital_bed', None),
    'pc_infectados': ('infectados', 'populacao'),
    'pc_isolados': ('in_isolation', 'populacao'),
    'pc_em_contagio': ('em_contagio', 'populacao'),
    'rt': ('transmitidos', 'contagio_finalizado'),
}
