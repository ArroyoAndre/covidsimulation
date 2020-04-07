## Probability of an outcome X or worse
# X = [NO_SYMPTOMS, MILD, MODERATE, SEVERE (hospitalization required), ITU, VENTILATION, DEATH]
#
# E.g.:  10-19's have a 0.6 (1-0.4) propability of not developing a detectable infection
#        30-39' have 0.241 probability of at least developing mild symptoms
#        80+ have 0.078 probability of death

OUTCOME_THRESHOLDS = [
    [0.3, 0.017, 0.0012, 0.000090, 0.000122, 0.000075, 0.000016],  # 0-9 years
    [0.4, 0.020, 0.003, 0.000400, 0.000365, 0.000245, 0.000070],  # 10-19
    [0.5, 0.105, 0.034, 0.011, 0.002, 0.001, 0.00031],  # 20-29
    [0.55, 0.241, 0.091, 0.034, 0.005, 0.003, 0.00084],  # 30-39
    [0.59, 0.225, 0.098, 0.043, 0.007, 0.005, 0.0016],  # 40-49
    [0.62, 0.286, 0.153, 0.082, 0.020, 0.015, 0.006],  # 50-59
    [0.65, 0.328, 0.197, 0.118, 0.047, 0.038, 0.019],  # 60-69
    [0.68, 0.339, 0.237, 0.166, 0.084, 0.072, 0.043],  # 70-79
    [0.763, 0.582, 0.327, 0.184, 0.161, 0.136, 0.078],  # 80+
]
