
class AgeGroup:
  def __init__(self, index, severity, isolation_adherence, isolation_effectiveness, 
               diagnosis_delay=None):
    self.index = index
    self.severity = severity
    self.isolation_adherence = isolation_adherence
    self.isolation_effectiveness = isolation_effectiveness
    self.diagnosis_delay = diagnosis_dealy  # If set, lab is ignored
