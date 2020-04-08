
class AgeGroup:
  def __init__(self, index, severity, isolation_adherence, isolation_effectiveness):
    self.index = index
    self.severity = severity
    self.isolation_adherence = isolation_adherence
    self.isolation_effectiveness = isolation_effectiveness
    self.diagnosis_delay = diagnosis_delay  # if set, lab is ignored
