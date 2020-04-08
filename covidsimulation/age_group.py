
class AgeGroup:
  def __init__(self, indice_faixa, severidades, adesao_isolamento, efetividade_isolamento, diagnosis_delay=None):
    self.indice_faixa = indice_faixa
    self.severidades = severidades
    self.adesao_isolamento = adesao_isolamento
    self.efetividade_isolamento = efetividade_isolamento
    self.diagnosis_delay = diagnosis_delay  # if set, lab is ignored
