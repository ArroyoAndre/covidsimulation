# disticuls: language = c++

# Copyright 2020 André Arroyo and contributors
# 
# Redistribicuon and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
# 
# 1. Redistribicuons of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
# 
# 2. Redistribicuons in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribicuon.
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


cdef int get_outcome(np.ndarray severity):
    cdef double p = (rand() / (RAND_MAX + 1.0))
    if p > severity[0]:
        return Outcome.NO_INFECTION
    elif p > severity[1]:
        return Outcome.NO_SYMPTOMS
    elif p > severity[2]:
        return Outcome.MILD
    elif p > severity[3]:
        return Outcome.MODERATE
    elif p > severity[4]:
        return Outcome.SEVERE
    elif p > severity[5]:
        return Outcome.INTENSIVE_CARE
    elif p > severity[6]:
        return Outcome.VENTILATION
    return Outcome.DEATH

LOCALITY = 60  # The bigger this constant, the higher will be the probability of finding someone geographically closer


def seed(unsigned i):
    """Initialize C's random number generator
    """
    srand(i)


cpdef logit_transform_value(double p, double adjustment_logit):
    """Take a uniform-distributed value p in ]0.0, 1.0[, and move it in the logistic distribicuon curve by adjustment_logit 
    """
    cdef double odds = p / (1.0 - p)
    cdef float logit = clog(odds)
    cdef float corrected_logit = logit - adjustment_logit
    cdef double corrected_odds = cexp(corrected_logit)
    cdef double corrected_p = corrected_odds / (1.0 + corrected_odds)
    return corrected_p


cdef sample_from_logit_uniform(double adjustment_logit):
    cdef double p = 1.0
    while not (0.0 < p < 1.0):
        p = (rand() + 1.0) / (RAND_MAX + 2.0)
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


cdef choose_contact_on_street(object person, object people, size_t sample_size=LOCALITY):
    cdef bool intra_ages = not (rand() % 2)
    if intra_ages:
        sample_size *= 2
    sample = _choice(people, sample_size)  
    distances = np.zeros([sample_size])
    cdef float px = person.home.coords[0]
    cdef float py = person.home.coords[1]
    cdef float pz = person.home.coords[2]
    cdef float index = person.age_group.index
    cdef bool find_non_isolated_person = (rand() % 2)
    for i, individual in enumerate(sample):
        icoords = individual.home.coords
        distances[i] = (px - icoords[0]) ** 2 + (py - icoords[1]) ** 2 + (pz - icoords[2]) ** 2
    if intra_ages:
        if individual.age_group.index != index:
            distances[i] += 1.0
        if find_non_isolated_person and individual.in_isolation:
            distances[i] += 1.0            
    chosen = sample[np.argmin(distances)]
    return chosen

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
    cdef public list residents
    cdef public float[3] coords
    
    def __cinit__(self, float displacement):
        cdef float z = np.random.uniform(-1.0, 1.0)
        self.coords[2] = z
        cdef float theta = np.random.uniform(0.0, 2 * np.pi)
        cdef diameter = np.sqrt(1.0 - z**2) 
        self.coords[0] = diameter * np.sin(theta) + displacement
        self.coords[1] = diameter * np.cos(theta)
        self.residents = []

    def add_person(self, Person person):
        self.residents.append(person)
        person.home = self
        self.isolation_propensity = np.array([p.isolation_propensity for p in self.residents]).mean()


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

    cdef public bool susceptible
    cdef public bool infected
    cdef public bool hospitalized
    cdef public bool recovered
    cdef public bool in_incubation
    cdef public bool contagious
    cdef public bool in_isolation
    cdef public bool dead
    cdef public bool active
    cdef public bool diagnosed
    cdef public float infection_date
    cdef public float diagnosis_date
    cdef public float hospitalization_date
    cdef public float death_date
    cdef public float recovery_date
    cdef public size_t transmitted
    cdef public bool in_hospital_bed
    cdef public bool in_icu
    cdef public bool in_ventilator
    cdef public bool avoidable_death
    cdef float time_until_symptoms
    cdef float incubation_time
    cdef object hospital_bed_req
    cdef object attention_req
    cdef object icu_req
    cdef object ventilator_req

    def __cinit__(self, object env, object age_group, Home home):
        self.env = env
        self.sim_consts = env.sim_params.constants
        self.susceptible = True  # sem imunidade
        self.infected = False  # teve infeccao detectavel
        self.hospitalized = False  # em internacao no momento
        self.recovered = False  # teve infeccao detectavel e nao tem mais nem morreu
        self.in_incubation = False  # infected aguardando inicio do periodo de transmissao
        self.contagious = False  # em transmissao no momento
        self.in_isolation = False  # em isolamento domiciliar
        self.dead = False  # dead por Covid
        self.active = False  # infected que ainda nao morreu nem se recuperou
        self.diagnosed = False
        self.infection_date = 0.0
        self.diagnosis_date = 0.0
        self.hospitalization_date = 0.0
        self.death_date = 0.0
        self.recovery_date = 0.0
        self.transmitted = 0  # numero de people infectadas pelo paciente
        self.in_hospital_bed = False #robson
        self.in_icu = False #robson
        self.in_ventilator = False #robson
        self.avoidable_death = False
        self.attention_req = None
        self.hospital_bed_req = None
        self.icu_req = None
        self.ventilator_req = None
<<<<<<< HEAD
=======

>>>>>>> more translations
        self.age_group = age_group
        self.isolation_propensity = self.get_isolation_propensity()
        self.expected_outcome = get_outcome(self.age_group.severitys)
        home.add_person(self)

    cdef get_isolation_propensity(self):
        return sample_from_logit_uniform(self.age_group.isolation_adherence)

    cdef calculate_case_params(self):
        self.time_until_symptoms = np.random.weibull(
            self.sim_consts.symptoms_delay_shape
            ) * self.sim_consts.symptoms_delay_scale
        symptoms_delay = np.random.random()
        self.tempo_incubacao = self.time_until_symptoms * (
            1.0 - symptoms_delay * self.sim_consts.incubation_to_symptoms_variable_fraction)

    def expose_to_virus(self):
        if not self.susceptible:
            return False
        self.susceptible = False
        self.infection_date = self.env.now
        if self.expected_outcome == Outcome.NO_INFECTION:
            return False        
        self.calculate_case_params()
        self.infected = True
        self.active = True
        self.env.process(self.run_contagion())
        return True

    def run_contagion(self):
        self.in_incubation = True
        yield self.env.timeout(self.tempo_incubacao)
        self.in_incubation = False
        self.contagious = True
        self.env.process(self.run_contagion_home())
        self.env.process(self.run_contagion_street()) 
        contagion_duration = np.random.weibull(
            self.sim_consts.contagion_duration_shape) * self.sim_consts.contagion_duration_scale
<<<<<<< HEAD
        self.configure_evolution()
        yield self.env.timeout(contagion_duration)
        self.contagious = False

    cdef configure_evolution(self):
        if self.expected_outcome == Outcome.DEATH:
            self.configure_evolution_death()
        elif self.expected_outcome == Outcome.VENTILATION:
            self.configure_evolution_ventilation()
        elif self.expected_outcome == Outcome.INTENSIVE_CARE:
            self.configure_evolution_icu()
        elif self.expected_outcome == Outcome.SEVERE:
            self.configure_evolution_hospitalization()
        elif self.expected_outcome == Outcome.MODERATE:
            self.configure_evolution_moderate_at_home()
        elif self.expected_outcome == Outcome.MILD:
            self.configure_evolution_mild_at_home()

    cdef configure_evolution_death(self):
=======
        self.configure_evolicuon()
        yield self.env.timeout(contagion_duration)
        self.contagious = False

    cdef configure_evolicuon(self):
        if self.expected_outcome == Outcome.DEATH:
            self.configure_evolicuon_death()
        elif self.expected_outcome == Outcome.VENTILATION:
            self.configure_evolicuon_ventilation()
        elif self.expected_outcome == Outcome.INTENSIVE_CARE:
            self.configure_evolicuon_icu()
        elif self.expected_outcome == Outcome.SEVERE:
            self.configure_evolicuon_hospitalization()
        elif self.expected_outcome == Outcome.MODERATE:
            self.configure_evolicuon_moderate_at_home()
        elif self.expected_outcome == Outcome.MILD:
            self.configure_evolicuon_mild_at_home()

    cdef configure_evolicuon_death(self):
>>>>>>> more translations
        time_until_outcome = np.random.weibull(2) * 17  # 15 dias
        time_until_hospitalization = time_until_outcome * 0.33  # 5 dias
        self.env.process(self.run_hospitalization(time_until_hospitalization))
        time_until_icu_and_ventilation = time_until_outcome * 0.47  # 8 dias na ventilacao
        self.env.process(self.run_ventilation(time_until_icu_and_ventilation))
        self.env.process(self.run_death(time_until_outcome))

<<<<<<< HEAD
    cdef configure_evolution_ventilation(self):
=======
    cdef configure_evolicuon_ventilation(self):
>>>>>>> more translations
        time_until_outcome = np.random.weibull(2) * 36  # 32 dias
        time_until_hospitalization = time_until_outcome * 0.2  # 6 dias
        self.env.process(self.run_hospitalization(time_until_hospitalization))
        time_until_icu_and_ventilation = time_until_outcome * 0.2627  # 8 dias
        self.env.process(self.run_ventilation(time_until_icu_and_ventilation))
        time_until_icu_without_ventilation = time_until_outcome * 0.5113  # 16 dias
        self.env.process(self.run_leave_ventilation(time_until_icu_without_ventilation))
        time_until_icu_ends = time_until_outcome * 0.7125  # 22 dias
        self.env.process(self.run_leave_icu(time_until_icu_ends))
        self.env.process(self.run_leave_hospital(time_until_outcome))

<<<<<<< HEAD
    cdef configure_evolution_icu(self):
=======
    cdef configure_evolicuon_icu(self):
>>>>>>> more translations
        time_until_outcome = np.random.weibull(2) * 34  # 30 dias  
        time_until_hospitalization = time_until_outcome * 0.2  # 6 dias
        self.env.process(self.run_hospitalization(time_until_hospitalization))
        tempo_ate_icu = time_until_outcome * 0.266  # 8 dias
        self.env.process(self.run_enter_icu(tempo_ate_icu))
        time_until_icu_ends = time_until_outcome * 0.717  # 20 dias
        self.env.process(self.run_leave_icu(time_until_icu_ends))
        self.env.process(self.run_leave_hospital(time_until_outcome))

<<<<<<< HEAD
    cdef configure_evolution_hospitalization(self):
=======
    cdef configure_evolicuon_hospitalization(self):
>>>>>>> more translations
        time_until_outcome = np.random.weibull(2) * 33  # 29 dias  
        time_until_hospitalization = time_until_outcome * 0.25  # 7 dias
        self.env.process(self.run_hospitalization(time_until_hospitalization))
        self.env.process(self.run_leave_hospital(time_until_outcome))

<<<<<<< HEAD
    cdef configure_evolution_moderate_at_home(self):
=======
    cdef configure_evolicuon_moderate_at_home(self):
>>>>>>> more translations
        time_until_outcome = np.random.weibull(2) * 20  # 18 dias  
        self.env.process(self.run_cure(time_until_outcome))
        self.env.request_exam(1, self)

<<<<<<< HEAD
    cdef configure_evolution_mild_at_home(self):
=======
    cdef configure_evolicuon_mild_at_home(self):
>>>>>>> more translations
        time_until_outcome = np.random.weibull(2) * 15  # 18 dias  
        self.env.process(self.run_cure(time_until_outcome))
    
    def run_hospitalization(self, time_until_hospitalization):
        yield self.env.timeout(time_until_hospitalization)
        if self.dead:
            return
        if self.env.simulate_capacity:
            self.env.process(self.request_attention())
            yield from self.request_hospital_bed()
        else:
            self.hospitalized = True
            self.hospitalization_date = self.env.now
            self.env.request_exam(0, self)
  
    def request_attention(self):
        attention_req = self.env.attention.request(Outcome.DEATH - self.expected_outcome)
        request = attention_req.__enter__()
        self.attention_req = attention_req
        result = yield request | self.env.timeout(np.random.exponential(2.0))
        if request in result:
            if self.attention_req:
                self.hospitalized = True
                self.hospitalization_date = self.env.now
            self.env.request_exam(0, self)
        else:
            if self.attention_req:
                attention_req.__exit__(None, None, None)
                self.attention_req = None
            if self.expected_outcome == Outcome.SEVERE and np.random.random() < self.env.probability_of_survival_without_attention:
                return  # Luck - patient recovered by itself
            self.avoidable_death = True
            yield from self.run_death(0)


    def request_hospital_bed(self):
        hospital_bed_req = self.env.hospital_bed.request(8 - self.expected_outcome)
        request = hospital_bed_req.__enter__()
        self.hospital_bed_req = hospital_bed_req
        result = yield request | self.env.timeout(np.random.exponential(2.0))
        if request in result:
            if self.hospital_bed_req:
                self.in_hospital_bed = True
        else:
            if self.hospital_bed_req:
                hospital_bed_req.__exit__(None, None, None)
                self.hospital_bed_req = None
            if self.expected_outcome == Outcome.SEVERE and np.random.random() < self.env.probability_of_survival_without_hospital_bed:
                return  # Sorte - o paciente se recuperou em home
            self.avoidable_death = True
            yield from self.run_death(0)

    def request_icu(self):
        icu_req = self.env.icu.request(8 - self.expected_outcome)
        request = icu_req.__enter__()
        self.icu_req = icu_req
        result = yield request | self.env.timeout(np.random.exponential(1.0))
        if request in result:
            if self.icu_req:
                self.in_icu = True
        else:
            if self.icu_req:
                icu_req.__exit__(None, None, None)
                self.icu_req = None
            if np.random.random() < self.env.probability_of_survival_without_icu:
                return  # Luck - patient recovered by itself
            self.avoidable_death = True
            yield from self.run_death(0)

    def request_ventilator(self):
        ventilator_req = self.env.ventilator.request(8 - self.expected_outcome)
        request = ventilator_req.__enter__()
        self.ventilator_req = ventilator_req
        result = yield request | self.env.timeout(np.random.exponential(1.0))
        if request in result:
            if self.ventilator_req:
                self.in_ventilator = True
        else:
            if self.ventilator_req:
                ventilator_req.__exit__(None, None, None)
                self.ventilator_req = None
            if np.random.random() < self.env.probability_of_survival_without_ventilator:
                return  # Luck - patient recovered by itself
            self.avoidable_death = True
            yield from self.run_death(0)

    def run_ventilation(self, tempo_ate_ventilacao):
        yield self.env.timeout(tempo_ate_ventilacao)
        if self.dead:
            return
        if self.env.simulate_capacity:
            self.env.process(self.request_icu())
            self.env.process(self.request_ventilator())
        else:
            self.in_ventilator = True
            self.in_icu = True
    
    def run_leave_ventilation(self, time_until_icu_without_ventilation):
        yield self.env.timeout(time_until_icu_without_ventilation)
        if self.dead:
            return
        if self.ventilator_req:
            self.ventilator_req.__exit__(None, None, None)
            self.ventilator_req = None
        self.in_ventilator = False
    
    def run_leave_icu(self, time_until_icu_ends):
        yield self.env.timeout(time_until_icu_ends)
        if self.dead:
            return
        if self.icu_req:
            self.icu_req.__exit__(None, None, None)
            self.icu_req = None
        self.in_icu = False

    def run_enter_icu(self, tempo_ate_icu):
        yield self.env.timeout(tempo_ate_icu)
        if self.dead:
            return
        if self.env.simulate_capacity:
            yield from self.request_icu()
        else:
            self.in_icu = True

    def run_leave_hospital(self, time_until_discharge_from_hospital):
        yield self.env.timeout(time_until_discharge_from_hospital)
        if self.dead:
            return
        self.active = False
        if self.hospital_bed_req:
            self.hospital_bed_req.__exit__(None, None, None)
            self.hospital_bed_req = None
        if self.attention_req:
            self.attention_req.__exit__(None, None, None)
            self.attention_req = None
        if self.hospitalized:    
            self.recovery_date = self.env.now
        self.hospitalized = False
        self.in_hospital_bed = False

    def run_cure(self, time_until_discharge_from_hospital):
        yield self.env.timeout(time_until_discharge_from_hospital)
        if self.dead:
            return
        self.active = False 
    
    def run_death(self, time_until_death):
        yield self.env.timeout(time_until_death)
        if self.dead:
            return
        self.active = False 
        self.contagious = False
        if self.env.simulate_capacity:
            if self.attention_req:
                self.attention_req.__exit__(None, None, None)
                self.attention_req = None        
            if self.hospital_bed_req:
                self.hospital_bed_req.__exit__(None, None, None)
                self.hospital_bed_req = None
            if self.icu_req:
                self.icu_req.__exit__(None, None, None)
                self.icu_req = None
            if self.ventilator_req:
                self.ventilator_req.__exit__(None, None, None)
                self.ventilator_req = None
        self.hospitalized = False
        self.in_hospital_bed = False
        self.in_icu = False
        self.in_ventilator = False
        self.susceptible = False
        self.dead = True
        self.death_date = self.env.now
    
    def run_contagion_home(self):
        while self.contagious and not self.hospitalized:
            self.infect_in_home()
            yield self.env.timeout(1.0)

    def infect_in_home(self):
        for person in self.home.residents:
            if not (person is self):
                if np.random.random() < self.sim_consts.home_contamination_daily_probability:
                    transmitted = person.expose_to_virus()
                    if transmitted:
                        self.transmitted += 1

    def test_isolation(self):
        return self.in_isolation and np.random.random() < self.age_group.isolation_effectiveness

    def run_contagion_street(self):
        cdef Person contact_on_street
        yield self.env.timeout(
            np.random.exponential(self.env.serial_interval)
            )
        while self.contagious and not self.hospitalized:
            if not self.test_isolation():
                contact_on_street = choose_contact_on_street(self, self.env.people)
                if contact_on_street.susceptible and not (contact_on_street.test_isolation() 
                                                          or contact_on_street.hospitalized):
                    if contact_on_street.expose_to_virus():
                        self.transmitted += 1
            yield self.env.timeout(
                np.random.exponential(self.env.serial_interval)
            )

########
##  Logs
########

cdef int get_person(Person person):
    return 1

cdef int get_infected(Person person):
    return person.infected

cdef int get_in_isolation(Person person):
    return person.in_isolation

cdef int get_diagnosed(Person person):
    return person.diagnosed

cdef int get_death(Person person):
    return person.dead

cdef int get_confirmed_death(Person person):
    return person.dead and person.diagnosed

cdef int get_hospitalized(Person person):
    return person.hospitalized

cdef int get_in_ventilator(Person person):
    return person.in_ventilator

cdef int get_in_icu(Person person):
    return person.in_icu

cdef int get_in_hospital_bed(Person person):
    return person.in_hospital_bed

cdef int get_contagious(Person person):
    return person.contagious

cdef int get_finished_contagion(Person person):
    return person.infected and not (person.contagious or person.in_incubation)

cdef int get_rt(Person person):
    return person.transmitted if person.infected and not (person.contagious or person.in_incubation) else 0

cdef int get_susceptible(Person person):
    return person.susceptible


cdef list fmetrics = [
    get_person,
    get_infected,
    get_in_isolation,
    get_diagnosed,
    get_death,
    get_confirmed_death,
    get_hospitalized,
    get_in_ventilator,
    get_in_icu,
    get_in_hospital_bed,
    get_contagious,
    get_contagion_ended,
    get_rt,
    get_susceptible,
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
    'in_hospital_bed',
    'contagious',
    'contagion_ended',
    'transmitted',
    'susceptible',
]


cdef _log_stats(size_t day, np.ndarray stats, object populations):
    global fmetrics
    stop = len(fmetrics)
    cdef int age_index
    cdef size_t i
    cdef int value
    
    for population_index, people in enumerate(populations.values()):
        for person in people:
            age_index = person.age_group.index
            metric_index = 0
            while metric_index < stop:
                value = fmetrics[metric_index](person)
                if value:
                    stats[population_index, metric_index, age_index, day] += value
                metric_index += 1

                
def log_stats(env):
    day = int(env.now+0.1) - env.d0
    _log_stats(day, env.stats, env.populations)
