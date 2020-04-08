# distutils: language = c++

# Copyright 2020 André Arroyo and contributors
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
# promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from libcpp cimport bool, float, int
import numpy as np
cimport numpy as np
from libc.math cimport log as clog, exp as cexp
from libc.stdlib cimport rand, RAND_MAX, srand
from libcpp.vector cimport vector
import simpy


cpdef enum Outcome:
    NO_INFECTION = 0
    NO_SYMPTOMS = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    INTENSIVE_CARE = 5
    VENTILATION = 6
    DEATH = 7


cpdef enum Age:
    A0_9 = 0
    A10_19 = 1
    A20_29 = 2
    A30_39 = 3
    A40_49 = 4
    A50_59 = 5
    A60_69 = 6
    A70_79 = 7
    A80plus = 8


age_str = [
    '0-9',
    '10-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80+',
]


cdef int get_outcome(np.ndarray severidade):
    cdef double p = (rand() / (RAND_MAX + 1.0))
    if p > severidade[0]:
        return Outcome.NO_INFECTION
    elif p > severidade[1]:
        return Outcome.NO_SYMPTOMS
    elif p > severidade[2]:
        return Outcome.MILD
    elif p > severidade[3]:
        return Outcome.MODERATE
    elif p > severidade[4]:
        return Outcome.SEVERE
    elif p > severidade[5]:
        return Outcome.INTENSIVE_CARE
    elif p > severidade[6]:
        return Outcome.VENTILATION
    return Outcome.DEATH

LOCALIDADE = 60  ## Quanto maior a localidade, maior a tendência de encontrar alguém geograficamente próximo


def seed(unsigned i):
    """Initialize C's random number generator
    """
    srand(i)


cpdef logit_transform_value(double p, double adjustment_logit):
    """Take a uniform-distributed value p in ]0.0, 1.0[, and move it in the logistic distribution curve by adjustment_logit 
    """
    cdef double odds = p / (1.0 - p)
    cdef float logit = clog(odds)
    cdef float corrected_logit = logit - adjustment_logit
    cdef double corrected_odds = cexp(corrected_logit)
    cdef double corrected_p = corrected_odds / (1.0 + corrected_odds)
    return corrected_p


cdef sample_from_logit_uniform(double adjustment_logit):
    cdef double p = (rand() + 1.0) / (RAND_MAX + 2.0)
    return logit_transform_value(p, adjustment_logit)


def choice(arr, sample_size):
    return _choice(arr, sample_size)


cdef sample_indices_with_replacement(size_t population_len, size_t sample_size):
    cdef vector[size_t] indices
    indices.reserve(sample_size)
    for _ in range(sample_size):
        indices.push_back(<size_t>(rand() / (RAND_MAX  + 1.0) * population_len))
    return indices


cdef _choice(list arr, size_t sample_size):
    return [arr[i] for i in sample_indices_with_replacement
(len(arr), sample_size)]


cdef escolher_contato_na_rua(object pessoa, object pessoas, size_t n_amostra=LOCALIDADE):
    cdef bool intra_idades = not (rand() % 2)
    if intra_idades:
        n_amostra *= 2
    amostra = _choice(pessoas, n_amostra)  
    distancias = np.zeros([n_amostra])
    cdef float px = pessoa.home.coords[0]
    cdef float py = pessoa.home.coords[1]
    cdef float pz = pessoa.home.coords[2]
    cdef float indice_faixa = pessoa.age_group.indice_faixa
    cdef bool encontra_pessoa_nao_isolada = (rand() % 2)
    for i, individuo in enumerate(amostra):
        icoords = individuo.home.coords
        distancias[i] = (px - icoords[0]) ** 2 + (py - icoords[1]) ** 2 + (pz - icoords[2]) ** 2
    if intra_idades:
        if individuo.age_group.indice_faixa != indice_faixa:
            distancias[i] += 1.0
        if encontra_pessoa_nao_isolada and individuo.em_isolamento:
            distancias[i] += 1.0            
    escolhido = amostra[np.argmin(distancias)]
    return escolhido

###
## SimulationConstants and defaults
###

cdef class SimulationConstants:
    cdef public float home_contamination_daily_probability
    cdef public float survival_probability_in_severe_overcapacity
    cdef public float survival_probability_without_hospital_bed
    cdef public float survival_probability_without_intensive_care_bed
    cdef public float survival_probability_without_ventilator
    cdef public float symptoms_delay_shape
    cdef public float symptoms_delay_scale
    cdef public float incubation_to_symptoms_variable_fraction
    cdef public float contagion_duration_shape
    cdef public float contagion_duration_scale

    def __init__(self):  # DEFAULT SIMULATION PARAMENTERS are set here
        self.home_contamination_daily_probability = 0.3
        self.survival_probability_in_severe_overcapacity = 0.3
        self.survival_probability_without_hospital_bed = 0.9
        self.survival_probability_without_intensive_care_bed = 0.8
        self.survival_probability_without_ventilator = 0.1
        self.symptoms_delay_shape = 4.0
        self.symptoms_delay_scale = 6.0
        self.incubation_to_symptoms_variable_fraction = 0.3
        self.contagion_duration_shape = 2.0
        self.contagion_duration_scale = 4.0




####
## Home - a place with geometric coordinates where people live
####

cdef class Home:
    cdef public float isolation_propensity
    cdef public list moradores
    cdef public float[3] coords
    
    def __cinit__(self, float displacement):
        cdef float z = np.random.uniform(-1.0, 1.0)
        self.coords[2] = z
        cdef float theta = np.random.uniform(0.0, 2 * np.pi)
        cdef diameter = np.sqrt(1.0 - z**2) 
        self.coords[0] = diameter * np.sin(theta) + displacement
        self.coords[1] = diameter * np.cos(theta)
        self.moradores = []

    def adicionar_pessoa(self, Person pessoa):
        self.moradores.append(pessoa)
        pessoa.home = self
        self.isolation_propensity = np.array([p.isolation_propensity for p in self.moradores]).mean()


####
## Person - an individual being simulated, with all characteristics about the disease
####


cdef class Person:
    cdef object env  # simpy Environment
    cdef public object age_group
    cdef SimulationConstants sim_consts
    cdef public float isolation_propensity
    cdef size_t expected_outcome
    cdef public Home home

    cdef public bool succeptible
    cdef public bool infectado
    cdef public bool internado
    cdef public bool recuperado
    cdef public bool em_incubacao
    cdef public bool em_contagio
    cdef public bool em_isolamento
    cdef public bool morto
    cdef public bool ativo
    cdef public bool diagnosticado
    cdef public float data_contagio
    cdef public float data_diagnostico
    cdef public float data_internacao
    cdef public float data_morte
    cdef public float data_recuperacao
    cdef public size_t transmitidos
    cdef public bool em_leito
    cdef public bool em_uti
    cdef public bool em_ventila_mec
    cdef public bool morte_evitavel
    cdef float tempo_ate_sintomas
    cdef float tempo_incubacao
    cdef object leito_req
    cdef object atencao_req
    cdef object uti_req
    cdef object ventilacao_req

    def __cinit__(self, object env, object age_group, Home home):
        self.env = env
        self.sim_consts = env.sim_params.constants
        self.succeptible = True  # sem imunidade
        self.infectado = False  # teve infeccao detectavel
        self.internado = False  # em internacao no momento
        self.recuperado = False  # teve infeccao detectavel e nao tem mais nem morreu
        self.em_incubacao = False  # infectado aguardando inicio do periodo de transmissao
        self.em_contagio = False  # em transmissao no momento
        self.em_isolamento = False  # em isolamento domiciliar
        self.morto = False  # morto por Covid
        self.ativo = False  # Infectado que ainda nao morreu nem se recuperou
        self.diagnosticado = False
        self.data_contagio = 0.0
        self.data_diagnostico = 0.0
        self.data_internacao = 0.0
        self.data_morte = 0.0
        self.data_recuperacao = 0.0
        self.transmitidos = 0  # numero de pessoas infectadas pelo paciente
        self.em_leito = False #robson
        self.em_uti = False #robson
        self.em_ventila_mec = False #robson
        self.morte_evitavel = False
        self.atencao_req = None
        self.leito_req = None
        self.uti_req = None
        self.ventilacao_req = None

        self.age_group = age_group
        self.isolation_propensity = self.get_isolation_propensity()
        self.expected_outcome = get_outcome(self.age_group.severidades)
        home.adicionar_pessoa(self)

    cdef get_isolation_propensity(self):
        return sample_from_logit_uniform(self.age_group.adesao_isolamento)

    cdef calcular_parametros_do_caso(self):
        self.tempo_ate_sintomas = np.random.weibull(
            self.sim_consts.symptoms_delay_shape
            ) * self.sim_consts.symptoms_delay_scale
        atraso_sintomas = np.random.random()
        self.tempo_incubacao = self.tempo_ate_sintomas * (
            1.0 - atraso_sintomas * self.sim_consts.incubation_to_symptoms_variable_fraction)

    def expose_to_virus(self):
        if not self.succeptible:
            return False
        self.succeptible = False
        self.data_contagio = self.env.now
        if self.expected_outcome == Outcome.NO_INFECTION:
            return False        
        self.calcular_parametros_do_caso()
        self.infectado = True
        self.ativo = True
        self.env.process(self.rodar_contaminacao())
        return True

    def rodar_contaminacao(self):
        self.em_incubacao = True
        yield self.env.timeout(self.tempo_incubacao)
        self.em_incubacao = False
        self.em_contagio = True
        self.env.process(self.rodar_contagio_casa())
        self.env.process(self.rodar_contagio_na_rua()) 
        contagion_duration = np.random.weibull(
            self.sim_consts.contagion_duration_shape) * self.sim_consts.contagion_duration_scale
        self.configurar_evolucao()
        yield self.env.timeout(contagion_duration)
        self.em_contagio = False

    cdef configurar_evolucao(self):
        if self.expected_outcome == Outcome.DEATH:
            self.configurar_evolucao_morte()
        elif self.expected_outcome == Outcome.VENTILATION:
            self.configurar_evolucao_ventilacao()
        elif self.expected_outcome == Outcome.INTENSIVE_CARE:
            self.configurar_evolucao_uti()
        elif self.expected_outcome == Outcome.SEVERE:
            self.configurar_evolucao_internacao()
        elif self.expected_outcome == Outcome.MODERATE:
            self.configurar_evolucao_moderado_em_casa()
        elif self.expected_outcome == Outcome.MODERATE:
            self.configurar_evolucao_leve_em_casa()

    cdef configurar_evolucao_morte(self):
        tempo_desfecho = np.random.weibull(2) * 17  # 15 dias
        tempo_ate_internacao = tempo_desfecho * 0.33  # 5 dias
        self.env.process(self.rodar_internacao(tempo_ate_internacao))
        tempo_ate_uti_e_ventilacao = tempo_desfecho * 0.47  # 8 dias na ventilacao
        self.env.process(self.rodar_ventilacao(tempo_ate_uti_e_ventilacao))
        self.env.process(self.rodar_morte(tempo_desfecho))

    cdef configurar_evolucao_ventilacao(self):
        tempo_desfecho = np.random.weibull(2) * 36  # 32 dias
        tempo_ate_internacao = tempo_desfecho * 0.2  # 6 dias
        self.env.process(self.rodar_internacao(tempo_ate_internacao))
        tempo_ate_uti_e_ventilacao = tempo_desfecho * 0.2627  # 8 dias
        self.env.process(self.rodar_ventilacao(tempo_ate_uti_e_ventilacao))
        tempo_ate_uti_sem_ventilacao = tempo_desfecho * 0.5113  # 16 dias
        self.env.process(self.rodar_sair_da_ventilacao(tempo_ate_uti_sem_ventilacao))
        tempo_ate_fim_uti = tempo_desfecho * 0.7125  # 22 dias
        self.env.process(self.rodar_sair_da_uti(tempo_ate_fim_uti))
        self.env.process(self.rodar_sair_do_hospital(tempo_desfecho))

    cdef configurar_evolucao_uti(self):
        tempo_desfecho = np.random.weibull(2) * 34  # 30 dias  
        tempo_ate_internacao = tempo_desfecho * 0.2  # 6 dias
        self.env.process(self.rodar_internacao(tempo_ate_internacao))
        tempo_ate_uti = tempo_desfecho * 0.266  # 8 dias
        self.env.process(self.rodar_entrar_na_uti(tempo_ate_uti))
        tempo_ate_fim_uti = tempo_desfecho * 0.717  # 20 dias
        self.env.process(self.rodar_sair_da_uti(tempo_ate_fim_uti))
        self.env.process(self.rodar_sair_do_hospital(tempo_desfecho))

    cdef configurar_evolucao_internacao(self):
        tempo_desfecho = np.random.weibull(2) * 33  # 29 dias  
        tempo_ate_internacao = tempo_desfecho * 0.25  # 7 dias
        self.env.process(self.rodar_internacao(tempo_ate_internacao))
        self.env.process(self.rodar_sair_do_hospital(tempo_desfecho))

    cdef configurar_evolucao_moderado_em_casa(self):
        tempo_desfecho = np.random.weibull(2) * 20  # 18 dias  
        self.env.process(self.rodar_curar(tempo_desfecho))
        self.request_diagnosis()

    cdef configurar_evolucao_leve_em_casa(self):
        tempo_desfecho = np.random.weibull(2) * 15  # 18 dias  
        self.env.process(self.rodar_curar(tempo_desfecho))

    def request_diagnosis(self):
        diagnosis_delay = self.age_group.diagnosis_delay
        if diagnosis_delay is None:
            self.env.solicitar_exame(0, self)
        else:
            self.env.process(self.wait_for_diagnosis(diagnosis_delay))

    def wait_for_diagnosis(self, float diagnosis_delay):
        cdef float time_for_diagnosis = np.random.weibull(4.0) * diagnosis_delay
        yield self.env.timeout(diagnosis_delay)
        self.diagnosticado = True

    def rodar_internacao(self, tempo_ate_internacao):
        yield self.env.timeout(tempo_ate_internacao)
        if self.morto:
            return
        if self.env.simula_capacidade:
            self.env.process(self.requisitar_atencao())
            yield from self.requisitar_leito()
        else:
            self.internado = True
            self.data_internacao = self.env.now
            self.request_diagnosis()
  
    def requisitar_atencao(self):
        atencao_req = self.env.atencao.request(Outcome.DEATH - self.expected_outcome)
        request = atencao_req.__enter__()
        self.atencao_req = atencao_req
        resultado = yield request | self.env.timeout(np.random.exponential(2.0))
        if request in resultado:
            if self.atencao_req:
                self.internado = True
                self.data_internacao = self.env.now
            self.request_diagnosis()
        else:
            if self.atencao_req:
                atencao_req.__exit__(None, None, None)
                self.atencao_req = None
            if self.expected_outcome == Outcome.SEVERE and np.random.random() < self.sim_consts.survival_probability_in_severe_overcapacity:
                return  # Sorte - o paciente se recuperou em home
            self.morte_evitavel = True
            yield from self.rodar_morte(0)

        
    def requisitar_leito(self):
        leito_req = self.env.leito.request(8 - self.expected_outcome)
        request = leito_req.__enter__()
        self.leito_req = leito_req
        resultado = yield request | self.env.timeout(np.random.exponential(2.0))
        if request in resultado:
            if self.leito_req:
                self.em_leito = True
        else:
            if self.leito_req:
                leito_req.__exit__(None, None, None)
                self.leito_req = None
            if self.expected_outcome == Outcome.SEVERE and np.random.random() < self.sim_consts.survival_probability_without_hospital_bed:
                return  # Sorte - o paciente se recuperou em home
            self.morte_evitavel = True
            yield from self.rodar_morte(0)

    def requisitar_uti(self):
        uti_req = self.env.uti.request(8 - self.expected_outcome)
        request = uti_req.__enter__()
        self.uti_req = uti_req
        resultado = yield request | self.env.timeout(np.random.exponential(1.0))
        if request in resultado:
            if self.uti_req:
                self.em_uti = True
        else:
            if self.uti_req:
                uti_req.__exit__(None, None, None)
                self.uti_req = None
            if np.random.random() < self.sim_consts.survival_probability_without_intensive_care_bed:
                return  # Sorte - o paciente se recuperou em home
            self.morte_evitavel = True
            yield from self.rodar_morte(0)

    def requisitar_ventilacao(self):
        ventilacao_req = self.env.ventilacao.request(8 - self.expected_outcome)
        request = ventilacao_req.__enter__()
        self.ventilacao_req = ventilacao_req
        resultado = yield request | self.env.timeout(np.random.exponential(1.0))
        if request in resultado:
            if self.ventilacao_req:
                self.em_ventila_mec = True
        else:
            if self.ventilacao_req:
                ventilacao_req.__exit__(None, None, None)
                self.ventilacao_req = None
            if np.random.random() < self.sim_consts.survival_probability_without_ventilator:
                return  # Sorte - o paciente se recuperou em home
            self.morte_evitavel = True
            yield from self.rodar_morte(0)

    def rodar_ventilacao(self, tempo_ate_ventilacao):
        yield self.env.timeout(tempo_ate_ventilacao)
        if self.morto:
            return
        if self.env.simula_capacidade:
            self.env.process(self.requisitar_uti())
            self.env.process(self.requisitar_ventilacao())
        else:
            self.em_ventila_mec = True
            self.em_uti = True
    
    def rodar_sair_da_ventilacao(self, tempo_ate_uti_sem_ventilacao):
        yield self.env.timeout(tempo_ate_uti_sem_ventilacao)
        if self.morto:
            return
        if self.ventilacao_req:
            self.ventilacao_req.__exit__(None, None, None)
            self.ventilacao_req = None
        self.em_ventila_mec = False
    
    def rodar_sair_da_uti(self, tempo_ate_fim_uti):
        yield self.env.timeout(tempo_ate_fim_uti)
        if self.morto:
            return
        if self.uti_req:
            self.uti_req.__exit__(None, None, None)
            self.uti_req = None
        self.em_uti = False

    def rodar_entrar_na_uti(self, tempo_ate_uti):
        yield self.env.timeout(tempo_ate_uti)
        if self.morto:
            return
        if self.env.simula_capacidade:
            yield from self.requisitar_uti()
        else:
            self.em_uti = True

    def rodar_sair_do_hospital(self, tempo_alta):
        yield self.env.timeout(tempo_alta)
        if self.morto:
            return
        self.ativo = False
        if self.leito_req:
            self.leito_req.__exit__(None, None, None)
            self.leito_req = None
        if self.atencao_req:
            self.atencao_req.__exit__(None, None, None)
            self.atencao_req = None
        if self.internado:    
            self.data_recuperacao = self.env.now
        self.internado = False
        self.em_leito = False

    def rodar_curar(self, tempo_alta):
        yield self.env.timeout(tempo_alta)
        if self.morto:
            return
        self.ativo = False 
    
    def rodar_morte(self, tempo_ate_morte):
        yield self.env.timeout(tempo_ate_morte)
        if self.morto:
            return
        self.ativo = False 
        self.em_contagio = False
        if self.env.simula_capacidade:
            if self.atencao_req:
                self.atencao_req.__exit__(None, None, None)
                self.atencao_req = None        
            if self.leito_req:
                self.leito_req.__exit__(None, None, None)
                self.leito_req = None
            if self.uti_req:
                self.uti_req.__exit__(None, None, None)
                self.uti_req = None
            if self.ventilacao_req:
                self.ventilacao_req.__exit__(None, None, None)
                self.ventilacao_req = None
        self.internado = False
        self.em_leito = False
        self.em_uti = False
        self.em_ventila_mec = False
        self.succeptible = False
        self.morto = True
        self.data_morte = self.env.now
    
    def rodar_contagio_casa(self):
        while self.em_contagio and not self.internado:
            self.contaminar_em_casa()
            yield self.env.timeout(1.0)

    def contaminar_em_casa(self):
        for pessoa in self.home.moradores:
            if not (pessoa is self):
                if np.random.random() < self.sim_consts.home_contamination_daily_probability:
                    transmitido = pessoa.expose_to_virus()
                    if transmitido:
                        self.transmitidos += 1
            #            print(self.env.now, 'Contagio em home')

    def testar_isolamento(self):
        return self.em_isolamento and np.random.random() < self.age_group.efetividade_isolamento

    def rodar_contagio_na_rua(self):
        cdef Person contato_na_rua
        yield self.env.timeout(
            np.random.exponential(self.env.tempo_medio_entre_contagios)
            )
        while self.em_contagio and not self.internado:
            if not self.testar_isolamento():
                contato_na_rua = escolher_contato_na_rua(self, self.env.pessoas)
                if contato_na_rua.succeptible and not (contato_na_rua.testar_isolamento() 
                                                    or contato_na_rua.internado):
                    if contato_na_rua.expose_to_virus():
            #          print(self.env.now, 'Contagio na rua')
                        self.transmitidos += 1
            yield self.env.timeout(
                np.random.exponential(self.env.tempo_medio_entre_contagios)
                )

########
##  Logs
########

cdef int get_pessoa(Person pessoa):
    return 1

cdef int get_infectados(Person pessoa):
    return pessoa.infectado

cdef int get_em_isolamento(Person pessoa):
    return pessoa.em_isolamento

cdef int get_diagnosticados(Person pessoa):
    return pessoa.diagnosticado

cdef int get_mortos(Person pessoa):
    return pessoa.morto

cdef int get_mortos_confirmados(Person pessoa):
    return pessoa.morto and pessoa.diagnosticado

cdef int get_internados(Person pessoa):
    return pessoa.internado

cdef int get_ventilados(Person pessoa):
    return pessoa.em_ventila_mec

cdef int get_em_uti(Person pessoa):
    return pessoa.em_uti

cdef int get_em_leito(Person pessoa):
    return pessoa.em_leito

cdef int get_em_contagio(Person pessoa):
    return pessoa.em_contagio

cdef int get_contagio_finalizado(Person pessoa):
    return pessoa.infectado and not (pessoa.em_contagio or pessoa.em_incubacao)

cdef int get_rt(Person pessoa):
    return pessoa.transmitidos if pessoa.infectado and not (pessoa.em_contagio or pessoa.em_incubacao) else 0

cdef int get_succeptible(Person pessoa):
    return pessoa.succeptible


cdef list fmetricas = [
    get_pessoa,
    get_infectados,
    get_em_isolamento,
    get_diagnosticados,
    get_mortos,
    get_mortos_confirmados,
    get_internados,
    get_ventilados,
    get_em_uti,
    get_em_contagio,
    get_contagio_finalizado,
    get_rt,
    get_succeptible,
    get_em_leito,
]


MEASUREMENTS = [
    'population',
    'infected',
    'in_isolation',
    'diagnosed',
    'deaths',
    'confirmed_deaths',
    'inpatients',
    'ventilated',
    'in_intensive_care',
    'contagious',
    'contagion_ended',
    'transmited',
    'succeptible',
    'in_hospital_bed',
]


cdef _log_estatisticas(size_t dia, np.ndarray stats, object populacoes):
    global fmetricas
    stop = len(fmetricas)
    cdef int indice_faixa
    cdef size_t i
    cdef int value
    
    for indice_populacao, pessoas in enumerate(populacoes.values()):
        for pessoa in pessoas:
            indice_faixa = pessoa.age_group.indice_faixa
            metric_index = 0
            while metric_index < stop:
                value = fmetricas[metric_index](pessoa)
                if value:
                    stats[indice_populacao, metric_index, indice_faixa, dia] += value
                metric_index += 1          

                
def log_estatisticas(env):
    dia = int(env.now+0.1) - env.d0
    _log_estatisticas(dia, env.stats, env.populacoes)
